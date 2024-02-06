# OCR a book that was scanned into a PDF

# Dependencies #
# installable: opencv-contrib-python==4.5.5.64, numpy, tesseract, PyMuPDF (fitz), scikit-learn, and their dependencies...
# 3 external files:  parabolic and rotation_spacing originally by github user endolith, transcode_monochrome from img2pdf by josch
# Optional: jbig2.exe from jbig2enc -- currently assumes user is running windows

from os import path, remove, getcwd
from sys import exit
from io import BytesIO
from tempfile import NamedTemporaryFile
from multiprocessing import Pool
from subprocess import Popen, PIPE
import argparse

import cv2
from PIL import Image

import numpy as np
import pytesseract
import fitz
from sklearn.mixture import GaussianMixture
from rotation_spacing import get_angle
from transcode_monochrome import transcode_monochrome

from tqdm import tqdm

########################################################
#                      image proc                     #
########################################################

def initializer(angle_range, skip_ocr, clustering,lang):
    global anglerange
    global skipOCR
    global useclustering
    global language
    anglerange=angle_range
    skipOCR=skip_ocr
    useclustering=clustering
    language=lang

# check if tesseract ocr returns an empty string -- easy blank page detection
def has_text(image):
    return bool(pytesseract.image_to_string(image, lang=language))

# fit gaussians to pixel distribution and get means assuming bimodal, uses difference of the means to detect empty pages
def GMM_means_test(image):
    imsmall = cv2.resize(image,(int(image.shape[1]/4),int(image.shape[0]/4)), interpolation=cv2.INTER_NEAREST) 
    lightdark=GaussianMixture(n_components=2).fit(imsmall.flatten().reshape(-1,1))
    imsmall=None
    meanslist= lightdark.means_.flatten()
    distance= np.abs(meanslist[0]-meanslist[1])
    return not (distance<30)     #if distance between means is <30, safe to assume the page is blank

#get dpi of page in document given its pagenumber, uses extracted image width and page cropbox width
def getdpi(pdf, pageno):
    pageimage=Image.open(BytesIO(pdf.extract_image(pdf[pageno].get_images()[0][0])['image']))
    return round(pageimage.width/pdf[0].cropbox.width*72)

#return dpi given the width of an image and the width of a pdf cropbox
def calcdpi(imagewidth, cropwidth):
    return round(imagewidth/cropwidth*72)

# noise removal
def remove_noise(image):
    return cv2.GaussianBlur(image,(5,5),0)

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

def not_empty(image):
    if useclustering:
        return GMM_means_test(image)
    else:
        return has_text(image)   

#thresholding
def thresholding(image):
    thresholded= cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if not_empty(thresholded):
        return thresholded
    else:
        return cv2.bitwise_or(thresholded, cv2.bitwise_not(thresholded))

#skew correction. takes an image in and returns an image rotated by an optimal angle within +/- anglerange
def deskew(image):
    if not (image-255).any(): #page is empty
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, get_angle(cv2.resize(image,center, interpolation=cv2.INTER_LANCZOS4), anglerange), 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


########################################################
#                       doc proc                       #
########################################################

# task for parallelization. input/output bytes of Document containing a single Page, which contains a single image
def do_OCR(pdfbytes):
    pagedoc=fitz.open(stream=pdfbytes,filetype='pdf')
    #extract image from page and do pre-OCR processing -- assumes one image per page
    img=Image.open(BytesIO(pagedoc.extract_image(pagedoc[0].get_images()[0][0])['image']))
    dpi=calcdpi(img.width,pagedoc[0].cropbox.width)
    img=img.convert('L').rotate(pagedoc[0].rotation) #convert to grayscale
    img=np.array(img)
    img=thresholding(deskew(img))

    #convert image to 1bpp Image
    binPIL=Image.frombytes(mode='1', size=img.shape[::-1], data=np.packbits(img, axis=1))

    #if jbig2.exe present perform an external compression step after OCR, 
    if path.exists(path.join(getcwd(),'jbig2.exe')):
        if not skipOCR:
	        pagepdf=pytesseract.image_to_pdf_or_hocr(binPIL,extension='pdf', config=f"--dpi {str(dpi)}", lang=language)
        else:
            pagepdf=pdfbytes
        #convert temporary png file to jbig2
        with NamedTemporaryFile(delete=False) as temppng:
            binPIL.save(temppng, format='PNG')
            temppng.close()
            jb2conv=Popen(['jbig2.exe', '-p', temppng.name], stdout=PIPE, stderr=PIPE)
            jb2out, stderr= jb2conv.communicate()
            remove(temppng.name)

        #replace image with jbig2
        jbig2pdf=fitz.open(stream=pagepdf,filetype='pdf')
        xref=jbig2pdf[0].get_images()[0][0]
        jbig2pdf.update_stream(xref, jb2out, compress=False)
        jbig2pdf.xref_set_key(xref,'BitsPerComponent','1')
        jbig2pdf.xref_set_key(xref,'ColorSpace',"/DeviceGray")
        jbig2pdf.xref_set_key(xref,'Filter',"/JBIG2Decode")
        jbig2pdf.xref_set_key(xref,'Decode',"[ 0 1 ]") # if you don't include this, the OCR images will be inverted
        if not skipOCR:
        	return jbig2pdf.tobytes()
        else:
        	return jbig2pdf.convert_to_pdf()

    else: # OCR with PIL's CCITT Group 4 compression
        if not skipOCR:
            tiff=BytesIO()
            binPIL.save(tiff, format='tiff',compression='group4')
            pagepdf=pytesseract.image_to_pdf_or_hocr(Image.open(tiff),extension='pdf', config=f"--dpi {str(dpi)}", lang=language)
            return pagepdf
        else:
            # pagedoc=fitz.open(stream=pagedoc.convert_to_pdf())
            xref=pagedoc[0].get_images()[0][0]
            pagedoc.update_stream(xref,transcode_monochrome(binPIL), compress=False)
            pagedoc.xref_set_key(xref,'BitsPerComponent','1')
            pagedoc.xref_set_key(xref,'ColorSpace',"/DeviceGray")
            pagedoc.xref_set_key(xref,'Filter',"/CCITTFaxDecode")
            pagedoc.xref_set_key(xref,'DecodeParms',f"<</Colors 1/BlackIs1 true/K -1/Columns {binPIL.width}/BitsPerComponent 1>>")
            return pagedoc.convert_to_pdf()


########################################################
#                       main code                      #
########################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='This script will attempt to deskew, compress, and OCR your book scans')

    # Input file
    parser.add_argument('input_file', help='Input PDF file for OCR')
    # Output file (optional)
    parser.add_argument('--output_file', help='Output file name - if not provided, it will be set to input_file + "_OCR"')
    # Language
    parser.add_argument('--lang', type=str, default='eng', help='Language selection for OCR. Options: '+str(pytesseract.get_languages(config='')))
    # Preserve Front Cover
    parser.add_argument('--front-cover', action='store_true', default=False, help='Pass the first page of the input PDF to the output unchanged. Default is False')
    # Preserve Back Cover
    parser.add_argument('--back-cover', action='store_true', default=False, help='Pass the final page of the input PDF to the output unchanged. Default is False')
    # Angle Range for Deskew Correction
    parser.add_argument('--angle-range', type=int, default=2, help='Deskewing: search for optimal angle between +/- angle_range degrees. Default is 2')
    # Skip OCR
    parser.add_argument('--skip-ocr', action='store_true', default=False, help='Perform deskewing and thresholding / compression only')
    # Use clustering for empty page detection
    parser.add_argument('--clustering', action='store_true', default=False, help='Use GMM clustering to detect empty pages')
    #metadata
    parser.add_argument('--title', type=str, default='', help='Set tags in metadata. Place input string in quotes.')
    parser.add_argument('--author', type=str, default='', help="''")
    parser.add_argument('--subject', type=str, default='', help="''")
    parser.add_argument('--keywords', type=str, default='', help="'' comma-separated")

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.skip_ocr:
        suff='_mono.pdf'
    else:
        suff='_OCR.pdf'
    # parse the arguments
    infile = args.input_file
    if infile.lower().endswith('.pdf'):
        infile=infile[:len(infile)-len('.pdf')]
    if not path.exists(infile+'.pdf'):
        print(f"Error: The file '{infile+'.pdf'}' does not exist")
        exit(1)
    if args.output_file:
    	outfile = args.output_file
    	if not outfile.lower().endswith('.pdf'):
    		outfile+='.pdf'
    else:
    	outfile = infile+suff
    if path.exists(outfile):
    	print(f"Error: The file '{outfile}' already exists")
    	proceed=input("Overwrite (Y/N?): ")
    	if not proceed.lower()=='y':
	        exit(1)
    frontcover = args.front_cover
    backcover = args.back_cover
    

    #import pdf, start output file generation
    indoc=fitz.open(infile+'.pdf')
    pagecount=indoc.page_count
    ocrfile=fitz.open()

    #cover image preservation handling
    if frontcover: # copy cover image unchanged
        pagecount-=1
        ocrfile.insert_pdf(indoc,0,0)     
    if backcover:
        pagecount-=1

    #extract pages into a list of fitz pdf objects and pickle them
    indocs=[fitz.open() for i in range(pagecount)]
    for i in range(pagecount):
        if frontcover:
            indocs[i].insert_pdf(indoc,i+1,i+1)
        else:
            indocs[i].insert_pdf(indoc,i,i)
    inbytes=[doc.tobytes() for doc in indocs]
    indocs=None

    #parallelize OCR task
    pool=Pool(initializer=initializer, initargs=(args.angle_range,args.skip_ocr,args.clustering, args.lang))
    outbytes=[]
    with tqdm(total=pagecount, desc="Processing", position=0) as pbar:
        for result in pool.imap(do_OCR, inbytes):
            outbytes.append(result)
            pbar.update()
        pool.close()
        pool.join()

    #write resulting single-page pdfs to a file
    outdocs= [fitz.open(stream=doc,filetype='pdf') for doc in outbytes]
    for doc in outdocs:
        ocrfile.insert_pdf(doc)

    if backcover: #copy back cover
        ocrfile.insert_pdf(indoc, pagecount-1, pagecount-1) 

    metadata=indoc.metadata
    if args.title:
        metadata['title']=args.title
    if args.author:
        metadata['author']=args.author
    if args.keywords:
        metadata['keywords']=args.keywords
    if args.subject:
        metadata['subject']=args.subject
    ocrfile.set_metadata(metadata)

    ocrfile.save(outfile, garbage=3)
    print(f"Saved result as {outfile}")

if __name__ == '__main__':
    main()
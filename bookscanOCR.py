# OCR a book that was scanned into a PDF

# Dependencies #
# installable: opencv-contrib-python==4.5.5.64, numpy, tesseract, PyMuPDF (fitz), scikit-learn, and their dependencies...
# 3 external files:  parabolic.py and rotation_spacing.py modified from endolith's code, transcode_monochrome.py from img2pdf by josch
# Optional: jbig2.exe from jbig2enc -- currently assumes user is running windows

from os import path, remove, getcwd
from sys import exit
import argparse
from io import BytesIO
from tempfile import NamedTemporaryFile
from multiprocessing import Pool
from subprocess import Popen, PIPE
from tqdm import tqdm
# import gc

import cv2
from PIL import Image

import numpy as np
import pytesseract
import fitz
from sklearn.mixture import GaussianMixture

from rotation_spacing import get_angle
from pageselect import selection
from transcode_monochrome import transcode_monochrome

########################################################
#                      image proc                     #
########################################################

def initializer(angle_range, skip_ocr, clustering,lang,dpi_cap):
    global anglerange
    global skipOCR
    global useclustering
    global language
    global dpicap
    anglerange=angle_range
    skipOCR=skip_ocr
    useclustering=clustering
    language=lang
    dpicap=dpi_cap

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

#return dpi given the width of an image and the width of a pdf cropbox
def calcdpi(imagewidth, cropwidth):
    return round(imagewidth/cropwidth*72)

#get dpi of page in document given its pagenumber, uses max image width and page cropbox width
def getdpi(pdf, pageno):
    imlist=pdf[pageno].get_images()
    width=np.max([imlist[n][2] for n in range(len(imlist))])
    return calcdpi(width,pdf[pageno].cropbox.width)

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
    # thresholded= cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY ,23,2)

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

# task for parallelization. input/output bytes of Document containing a single Page
def do_OCR(pdfbytes):
    try:
        pagedoc=fitz.open(stream=pdfbytes,filetype='pdf')
        
        #dpi setting
        ogdpi=getdpi(pagedoc,0)
        dpi = dpicap if (dpicap and dpicap>72 and ogdpi>dpicap) else ogdpi

        #extract image from page
        im_infolist=pagedoc[0].get_images()
        if len(im_infolist)>1:
        	#rasterize page that contains more than one image
            img=Image.open(BytesIO(pagedoc[0].get_pixmap(dpi=dpi).tobytes()))
        elif len(im_infolist)==1:
            img=Image.open(BytesIO(pagedoc.extract_image(im_infolist[0][0])['image']))
            if dpi!=ogdpi:
                scaling_factor=dpi/ogdpi
                img.thumbnail((img.size[0]*scaling_factor,img.size[1]*scaling_factor))
        else:
            raise ValueError("No images found in the PDF page.")

        #grayscale and rotate with PIL, deskew and threshold with cv2
        img=img.convert('L').rotate(pagedoc[0].rotation)
        imarr=thresholding(deskew(np.array(img)))
        del img
        #convert cv2 image to 1bpp Image
        binPIL=Image.frombytes(mode='1', size=imarr.shape[::-1], data=np.packbits(imarr, axis=1))
        del imarr

        #if jbig2.exe present perform an external compression step after OCR, 
        if path.exists(path.join(getcwd(),'jbig2.exe')):
            if not skipOCR:
    	        pagepdf=pytesseract.image_to_pdf_or_hocr(binPIL,extension='pdf', config=f"--dpi {str(dpi)}", lang=language)
            else:
                with BytesIO() as PILpdf:
                    binPIL.save(PILpdf,format='PDF',resolution=dpi)
                    pagepdf=fitz.open(stream=PILpdf,filetype="pdf").convert_to_pdf()
            
            #convert temporary png file to jbig2
            with NamedTemporaryFile(delete=False) as temppng:
                try:
                    binPIL.save(temppng, format='PNG')
                    temppng.close()
                    jb2conv = Popen(['jbig2.exe', '-p', temppng.name], stdout=PIPE, stderr=PIPE)
                    jb2out, stderr = jb2conv.communicate(timeout=60)
                    if jb2conv.returncode != 0:
                        raise RuntimeError(f"jbig2.exe error: {stderr.decode()}")
                finally:
                    remove(temppng.name)

            #replace image with jbig2
            with fitz.open(stream=pagepdf, filetype='pdf') as jbig2pdf:
                del pagepdf
                xref=jbig2pdf[0].get_images()[0][0]
                jbig2pdf.update_stream(xref, jb2out, compress=False)
                del jb2out
                jbig2pdf.xref_set_key(xref,'BitsPerComponent','1')
                jbig2pdf.xref_set_key(xref,'ColorSpace',"/DeviceGray")
                jbig2pdf.xref_set_key(xref,'Filter',"/JBIG2Decode")
                jbig2pdf.xref_set_key(xref,'Decode',"[ 0 1 ]") # if you don't include this, the OCR images will be inverted
                return jbig2pdf.tobytes() if not skipOCR else jbig2pdf.convert_to_pdf()

        else: # OCR with PIL's CCITT Group 4 compression
            if not skipOCR:
                with BytesIO() as tiff:
                    binPIL.save(tiff, format='tiff',compression='group4')
                    return pytesseract.image_to_pdf_or_hocr(Image.open(tiff),extension='pdf', config=f"--dpi {str(dpi)}", lang=language)
            else:
                with BytesIO() as PILpdf:
                    binPIL.save(PILpdf,format='PDF',resolution=dpi)
                    with fitz.open(stream=PILpdf,filetype="pdf") as tifpdf:
                        xref=tifpdf[0].get_images()[0][0]
                        tifpdf.update_stream(xref,transcode_monochrome(binPIL), compress=False)
                        tifpdf.xref_set_key(xref,'BitsPerComponent','1')
                        tifpdf.xref_set_key(xref,'ColorSpace',"/DeviceGray")
                        tifpdf.xref_set_key(xref,'Filter',"/CCITTFaxDecode")
                        tifpdf.xref_set_key(xref,'DecodeParms',f"<</Colors 1/BlackIs1 true/K -1/Columns {binPIL.width}/BitsPerComponent 1>>")
                        return tifpdf.convert_to_pdf()
    finally:
        del pagedoc
        del binPIL
        # gc.collect()
########################################################
#                        parser                        #
########################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='This script will attempt to deskew, compress, and OCR your book scans')

    # Input file
    parser.add_argument('input_file', help='Input PDF file for OCR')
    # Output file (optional)
    parser.add_argument('--output_file', help='Output file name - if not provided, it will be set to input_file + "_OCR"')
    # Page Intervals
    parser.add_argument('--page_intervals', default=None, help='Comma-separated string of page numbers to keep in selection -- e.g. "3,5-7,21"')
    # Language
    parser.add_argument('--lang', type=str, default='eng', help='Language selection for OCR. Options: '+str(pytesseract.get_languages(config='')))
    # Preserve Front Cover
    parser.add_argument('--front-cover', action='store_true', default=False, help='Pass first page of input PDF to output unchanged. Applies after page_intervals')
    # Preserve Back Cover
    parser.add_argument('--back-cover', action='store_true', default=False, help='Pass final page of selection to output unchanged. Default is False, as above')
    # Angle Range for Deskew Correction
    parser.add_argument('--angle-range', type=int, default=2, help='Deskewing: search for optimal angle between +/- angle_range degrees. Default: 2')
    # Skip OCR
    parser.add_argument('--skip-ocr', action='store_true', default=False, help='Perform deskewing and thresholding / compression only')
    # DPI cap
    parser.add_argument('--dpi-cap', type=int, default=None, help='Output PDF resolution matches input unless greater than cap. Default is None')
    # Use clustering for empty page detection
    parser.add_argument('--clustering', action='store_true', default=False, help='Use GMM clustering to detect empty pages')
    #metadata
    parser.add_argument('--title', type=str, default='', help='Set tags in metadata. Place input string in quotes.')
    parser.add_argument('--author', type=str, default='', help="''")
    parser.add_argument('--subject', type=str, default='', help="''")
    parser.add_argument('--keywords', type=str, default='', help="'' comma-separated")

    return parser.parse_args()

########################################################
#                      generator                       #
########################################################

#extract page from a fitz pdf object and pickle it. frontcover & backcover are booleans
def generate_inbytes(doc, frontcover,backcover):
    fc=int(frontcover)
    for i in range(doc.page_count-fc-int(backcover)):
        pagedoc=fitz.open()
        pagedoc.insert_pdf(doc,i+fc,i+fc)
        docbytes=pagedoc.tobytes()
        yield docbytes
        pagedoc.close()
        del pagedoc
        del docbytes

########################################################
#                      main code                       #
########################################################

def main():
    args = parse_arguments()

    # Determine file suffix based on OCR flag
    suff = '_mono.pdf' if args.skip_ocr else '_OCR.pdf'

    # Input file processing
    infile = args.input_file
    if infile.lower().endswith('.pdf'):
        infile = infile[:len(infile) - len('.pdf')]
    if not path.exists(infile + '.pdf'):
        print(f"Error: The file '{infile + '.pdf'}' does not exist")
        exit(1)

    # Output file handling
    outfile = args.output_file or infile + suff
    if not outfile.lower().endswith('.pdf'):
        outfile += '.pdf'
    if path.exists(outfile):
        print(f"Error: The file '{outfile}' already exists")
        proceed = input("Overwrite (Y/N?): ")
        if not proceed.lower() == 'y':
            exit(1)

    # Front and back cover options
    frontcover, backcover = args.front_cover, args.back_cover
    
    #import pdf, select pages
    indoc=fitz.open(infile+'.pdf')
    if isinstance(args.page_intervals, str):
        indoc=selection(args.page_intervals,indoc)
    pagecount=indoc.page_count

    # start output file generation
    ocrfile=fitz.open()
    #cover image preservation handling
    if frontcover: # copy cover image unchanged
        pagecount-=1
        ocrfile.insert_pdf(indoc,0,0)     
    if backcover:
        pagecount-=1

    #parallelize OCR task
    with Pool(initializer=initializer, initargs=(args.angle_range, args.skip_ocr, args.clustering, args.lang,args.dpi_cap)) as pool:
        with tqdm(total=pagecount, desc="Processing", position=0) as pbar:
            for result in pool.imap(do_OCR, generate_inbytes(indoc,frontcover,backcover)):
                respdf=fitz.open(stream=result, filetype='pdf')
                ocrfile.insert_pdf(respdf)
                respdf.close()
                del respdf
                del result
                pbar.update()
            pool.close()
            pool.join()

    if backcover: #copy back cover
        ocrfile.insert_pdf(indoc, pagecount-1, pagecount-1) 

    # update metadata
    metadata=indoc.metadata
    for key, value in vars(args).items():
        if key in ['title', 'author', 'keywords', 'subject'] and value:
            metadata[key] = value
    ocrfile.set_metadata(metadata)

    # save result
    try:
        ocrfile.save(outfile, garbage=3)
        print(f"Saved result as {outfile}")
    except:
        print(f"Permission denied for filename {outfile}")
        # Generate a new file name by appending a timestamp
        from time import time
        new_outfile = f"{outfile[:-4]}_{int(time())}.pdf"
        try:
            ocrfile.save(new_outfile, garbage=3)
            print(f"PDF saved successfully with a new name: {new_outfile}")
        except:
            print(f"Failed to save pdf even with new name")
            exit(1)

if __name__ == '__main__':
    main()
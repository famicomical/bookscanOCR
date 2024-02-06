from io import BytesIO
from PIL import Image, TiffImagePlugin

# temporary change the attribute of an object using a context manager
class temp_attr:
    def __init__(self, obj, field, value):
        self.obj = obj
        self.field = field
        self.value = value

    def __enter__(self):
        self.exists = False
        if hasattr(self.obj, self.field):
            self.exists = True
            self.old_value = getattr(self.obj, self.field)
        # logger.debug(f"setting {self.obj}.{self.field} = {self.value}")
        setattr(self.obj, self.field, self.value)

    def __exit__(self, exctype, excinst, exctb):
        if self.exists:
            setattr(self.obj, self.field, self.old_value)
        else:
            delattr(self.obj, self.field)

def ccitt_payload_location_from_pil(img):
    # If Pillow is passed an invalid compression argument it will ignore it;
    # make sure the image actually got compressed.
    if img.info["compression"] != "group4":
        raise ValueError(
            "Image not compressed with CCITT Group 4 but with: %s"
            % img.info["compression"]
        )

    # Read the TIFF tags to find the offset(s) of the compressed data strips.
    strip_offsets = img.tag_v2[TiffImagePlugin.STRIPOFFSETS]
    strip_bytes = img.tag_v2[TiffImagePlugin.STRIPBYTECOUNTS]

    # PIL always seems to create a single strip even for very large TIFFs when
    # it saves images, so assume we only have to read a single strip.
    # A test ~10 GPixel image was still encoded as a single strip. Just to be
    # safe check throw an error if there is more than one offset.
    if len(strip_offsets) != 1 or len(strip_bytes) != 1:
        raise NotImplementedError(
            "Transcoding multiple strips not supported by the PDF format"
        )

    (offset,), (length,) = strip_offsets, strip_bytes

    # logger.debug("TIFF strip_offsets: %d" % offset)
    # logger.debug("TIFF strip_bytes: %d" % length)

    return offset, length

def transcode_monochrome(imgdata):
    """Convert the open PIL.Image imgdata to compressed CCITT Group4 data"""

    # logger.debug("Converting monochrome to CCITT Group4")

    # Convert the image to Group 4 in memory. If libtiff is not installed and
    # Pillow is not compiled against it, .save() will raise an exception.
    newimgio = BytesIO()

    # we create a whole new PIL image or otherwise it might happen with some
    # input images, that libtiff fails an assert and the whole process is
    # killed by a SIGABRT:
    #   https://gitlab.mister-muffin.de/josch/img2pdf/issues/46
    im = Image.frombytes(imgdata.mode, imgdata.size, imgdata.tobytes())

    # Since version 8.3.0 Pillow limits strips to 64 KB. Since PDF only
    # supports single strip CCITT Group4 payloads, we have to coerce it back
    # into putting everything into a single strip. Thanks to Andrew Murray for
    # the hack.
    #
    # Since version 8.4.0 Pillow allows us to modify the strip size explicitly
    tmp_strip_size = (imgdata.size[0] + 7) // 8 * imgdata.size[1]
    if hasattr(TiffImagePlugin, "STRIP_SIZE"):
        # we are using Pillow 8.4.0 or later
        with temp_attr(TiffImagePlugin, "STRIP_SIZE", tmp_strip_size):
            im.save(newimgio, format="TIFF", compression="group4")
    else:
        # only needed for Pillow 8.3.x but works for versions before that as
        # well
        pillow__getitem__ = TiffImagePlugin.ImageFileDirectory_v2.__getitem__

        def __getitem__(self, tag):
            overrides = {
                TiffImagePlugin.ROWSPERSTRIP: imgdata.size[1],
                TiffImagePlugin.STRIPBYTECOUNTS: [tmp_strip_size],
                TiffImagePlugin.STRIPOFFSETS: [0],
            }
            return overrides.get(tag, pillow__getitem__(self, tag))

        with temp_attr(
            TiffImagePlugin.ImageFileDirectory_v2, "__getitem__", __getitem__
        ):
            im.save(newimgio, format="TIFF", compression="group4")

    # Open new image in memory
    newimgio.seek(0)
    newimg = Image.open(newimgio)

    offset, length = ccitt_payload_location_from_pil(newimg)

    newimgio.seek(offset)
    return newimgio.read(length)
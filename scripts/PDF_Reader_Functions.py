from pytesseract import image_to_string, image_to_osd, TesseractError
import cv2
import os
import tempfile
from pdf2image import convert_from_path
import numpy as np
import os
import imutils
import time
import math
import re
from PDF_Reader.swt import SWTScrubber

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
    startY = 0 #int(image.shape[0] * 0.67) #0
    endY = int(image.shape[0]) # int(image.shape[0]*0.46)

    for y in range(startY, endY, stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def morphImg(image):
    blur = cv2.blur(image, (5, 5))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 0, 100, apertureSize=3)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=2)
    cv2.imshow('edges', edges)
    minLineLength = 3000
    maxLineGap = 100
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    # print(lines)
    for l in lines:
        cv2.line(image, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 0), 2)

    cv2.imshow('houghlines', image)

def resizeImg(image, scalePercent):
    # resizeImg
    imgCopy = image.copy()
    # scale_percent = 40  # percent of original size
    width = int(image.shape[1] * scalePercent / 100)
    height = int(image.shape[0] * scalePercent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image

def convertPDFtoImgs(filenamePDF, save_dir):

    with tempfile.TemporaryDirectory() as path:
        images_from_path = convert_from_path(filenamePDF, output_folder=save_dir)#, last_page=1, first_page=0)

    base_filename = os.path.splitext(os.path.basename(filenamePDF))[0]
    paths = []
    for i, page in enumerate(images_from_path):
        path = save_dir + base_filename + "-" + str(i) + ".jpg"
        page.save(path, 'JPEG')
        img = cv2.imread(path)
        checkOrientation(img, path)
        paths.append(path)

    # Clean up *ppm files created from process
    test = os.listdir(save_dir)
    for item in test:
        if item.endswith(".ppm"):
            os.remove(os.path.join(save_dir, item))
    return paths

def cleanImgs(imgFiles, kernel):
    # Clean and Enhance PDF page images
    cleanedImgs = []
    for i, img in enumerate(imgFiles):
        cleanedImg = cv2.imread(img)
        #print(type(cleanedImg))
        #print(np.mean(cleanedImg, axis=(0, 1))[0])
        #print(np.mean(cleanedImg.all()))
        #cv2.imshow("cleanedImg", cleanedImg)
        #cv2.waitKey()
        #blur = cv2.blur(img, (5, 5))
        #gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        #ret, cleanedImg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        #cleanedImg = cv2.dilate(cleanedImg, kernel, iterations=2)
        cleanedImgs.append(cleanedImg)

    return cleanedImgs

def collectData(imgs, searchedTerms):
    data = []
    for i, img in enumerate(imgs):
        print("Image ", i)
        (winW, winH) = (int(img.shape[1]*.3), int(img.shape[0]*.04))
        #if i < 1: # skip img 'n'
           #continue
        # loop over the image pyramid

        termFound = False
        counter = 0
        for (x, y, window) in sliding_window(img, stepSize=50, windowSize=(winW, winH)):

            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # startTime = time.time()
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # since we do not have a classifier, we'll just draw the window
            clone = img.copy()
            crop_img = clone[y:y + winH, x:x + winW]
            ret, crop_img = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY)
            #kernel = np.ones((1, 1), np.uint8)
            #crop_img = cv2.dilate(crop_img, kernel, iterations=2)
            #mean = np.mean(crop_img.all())
            mean = np.mean(crop_img, axis=(0, 1))[0]
            #print(mean)
            #mean, stddev = cv2.meanStdDev(cv2.UMat(crop_img.all()))


            #cv2.waitKey(1000)
            # IGNORE BLACK IMAGES
            if mean < 247 and mean > 180:

                #temp = resizeImg(clone, 40)#scalePercent=40)
                #cv2.imshow("clone", clone)
                #cv2.waitKey(100)
                #clone = resizeImg(clone, scalePercent=40)
                #crop_img = resizeImg(crop_img, 1.5)

                if termFound or counter%3 == 0:
                    crop_img = resizeImg(crop_img, 1.5)
                    output = image_to_string(crop_img, lang='eng')
                    #print(output)
                    #cv2.waitKey(50)
                    termFound = False
                    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                    #cv2.imshow("cropped", crop_img)
                    #cv2.imshow("Window", resizeImg(clone, imgScale=0.5))
                    #cv2.waitKey(50)
                    tempNum = "0"
                    try:
                        tempNum = re.sub("\D", "", output)  # str(int(filter(str.isdigit, output)))
                    except:
                        TypeError
                    termFound = False
                    if len(tempNum) > 2:
                        for j, term in enumerate(searchedTerms):
                            if term in output:
                                #print("Term was Found! Term = ", term, ", output = ", output)
                                output = improveImgQuality(crop_img, output, term, searchedTerms)
                                data.append(output)
                                termFound = True
                        #cv2.waitKey()

            #if wordToSearch in output:
                #print("Searched string found!! Output = ", output, "wordToSearch = ", wordToSearch)
            # endTime = time.time()
            # print("Time taken = ", endTime - startTime)
            counter = counter + 1

            # time.sleep(0.00001)

    return data

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches


'''https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033
    https://docs.opencv.org/3.3.0/d7/d4d/tutorial_py_thresholding.html
    '''
def improveImgQuality(crop_img, output, term, searchedTerms):

    outputList = []
    otherTermFound = True
    if term == "Code":
        otherTermFound = False
        #print("Code in output!")
        for j, termSearch in enumerate(searchedTerms):
            if termSearch in output and termSearch != "Code":
                otherTermFound = True
                #print("Other Term Found")
                term = termSearch
                break

    if not otherTermFound:
        #print("Still No Other Term Found")
        #term = "termUnknown"
        kernel = np.ones((2, 2), np.uint8)
        outputOld = output
        #print("outputOld = ", outputOld)
        crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)
        crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE, kernel)
        crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)
        crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE, kernel)
        output = image_to_string(crop_img, lang='eng')

        #print("newOutput2 = ", output)
        for j, termSearch in enumerate(searchedTerms):
            if termSearch in output and termSearch != "Code":
                #otherTermFound = True
                term = termSearch
                #print("Other word besides output found after morphological operations! term = ", term)
                locTerm = output.find("Code")
                if term != "Code":
                    outputOld = outputOld[:locTerm] + term + outputOld[locTerm:]
                outputList.append(outputOld)
                outputList.append(output)
                break

    else:
        #nlines = output.count('\n')
        locTerm = output.find(term)
        #print(loc)
        #print(len(output))
        #print(nlines)

        #print("INITIAL OUTPUT: ", output, "\n______________________________\n")
        outputList.append(output)
        nlinesList = list(find_all(output, '\n'))
        #print(nlinesList)
        nLineIndexMax = 0
        lineBeforeLoc = 0
        lineAfterLoc = len(output)
        nextLineFound = False
        fracHeightImg = 0
        fracWidthImg = 0

        for i in range (0, len(nlinesList)):
            if nlinesList[i] < locTerm and i > 0:
                nLineIndexMax = i
                lineBeforeLoc = nlinesList[i]
            elif not nextLineFound:
                lineAfterLoc = nlinesList[i]
                nextLineFound = True

        #cv2.imshow("crop_img_before", crop_img)
        if len(nlinesList) > 0:
            fracHeightImg = nLineIndexMax/len(nlinesList)

        if lineAfterLoc > 0:
            fracWidthImg = (locTerm - lineBeforeLoc) / lineAfterLoc

        if fracWidthImg> 0.2:
            fracWidthImg = 0.2

        #fracHeightImg = np.min(fracHeightImg*, 0.5) # Restrict to half-height
        if fracHeightImg > 0.5:
            fracHeightImg = 0.5
        #fracHeightImg = np.iinfo([fracHeightImg, 0.5]).min
        #outputOld = output
        #output = output[lineBeforeLoc: len(output)]
        #diffLen = len(outputOld) - len(output)
        #print(output)
        #print(outputOld[lineBeforeLoc: len(output)])

        startY = int(fracHeightImg*crop_img.shape[0])
        startX = int(fracWidthImg*crop_img.shape[1])
        crop_img = crop_img[startY:crop_img.shape[0], startX:crop_img.shape[1]]
        output = image_to_string(crop_img, lang='eng')
        #print("CROPPED OUTPUT BEFORE MORPHOLOGICAL PROCESS: ", output, "\n______________________________\n")
        outputList.append(term + output)
        kernel = np.ones((2, 2), np.uint8)

        crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)
        crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE, kernel)
        crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)
        crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE, kernel)


        #crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)
        #crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE, kernel)


        output = image_to_string(crop_img, lang='eng')
        #print("MORPHOLOGICAL OPERATIONS OUTPUT: ", output, "\n______________________________\n")
        outputList.append(term + output)
        #crop_img = crop_img[0:y + winH, x:x + winW]

        #ret, cleanedImg = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY)
        #kernel = np.ones((1, 1), np.uint8)
        #filtered = cv2.adaptiveThreshold(crop_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
        #opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        #img = image_smoothening(crop_img)
        #or_image = cv2.bitwise_or(img, closing)
        #cleanedImg = cv2.morphologyEx(cleanedImg, cv2.MORPH_CLOSE, kernel, iterations=5)
        #cleanedImg = cv2.dilate(cleanedImg, kernel, iterations=1)
        #cv2.imshow("image_smoothening", img)
        #cv2.imshow("or_image", or_image)

        #cv2.imshow("crop_img", crop_img)
        #cv2.imshow("cleanedImg", cleanedImg)
        #cv2.waitKey()
    return outputList

def returnDataFile(datafile, load_dir, save_dir):
    filenamePDF = load_dir + datafile + ".pdf"
    return filenamePDF

def cleanImgCollectData(imgFile, dilateKernel, dataToSearch):
    cleanedImgs = cleanImgs(imgFile, dilateKernel)
    data = collectData(cleanedImgs, dataToSearch)
    writeDataToFile(data)

def writeDataToFile(data):
    if len(data):
        savedData = open("savedData.txt", "w")
        for j in range(0, len(data)):
            data[j] = (data[j]).replace("\n", "<<<<<<") # remove new lines in string and replace with "<<<<<<" # Avoid numbers on new line from being appebded
            savedData.write(data[j] + "\n")

        savedData.close()

def checkNumbersVertically(x, y, pdfImg):
    return 0

'''____________________ORIENTATION AND SKEWNESS CORRECTION____________________'''

def checkOrientation(img, path):

    print("Processing PDF Page: ", path)
    errorIncurred = False

    try:
        pageDescrp = image_to_osd(img)
    except TesseractError:
        print("Tesseract Orientation Error")
        errorIncurred = True

    if not errorIncurred:

        index1 = pageDescrp.find('Rotate')
        index2 = pageDescrp.find('confidence')
        orienDescrp = pageDescrp[index1:index2]
        rotateAngle = list(filter(str.isdigit, orienDescrp))
        orienDescrp = ""
        for i in range(0, len(rotateAngle)):
            orienDescrp = orienDescrp + rotateAngle[i]

        # Perform Rotation if required
        rotateAngle = int(orienDescrp)

        if rotateAngle > 0:
            rotatedImg = imutils.rotate_bound(img, rotateAngle)
            cv2.imwrite(path, rotatedImg)

        else:  # Check Angular Rotation of Image:
            checkSkewness(img, path)

    else: # Check Angular Rotation of Image:
        checkSkewness(img, path)



def resizeImg(img, imgScale):
    #print("img.shape[1] = ", img.shape[1])
    #print("img.shape[0] = ", img.shape[0])
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    newimg = cv2.resize(img, (int(newX), int(newY)))
    return newimg

def checkHorizontalSkew(img, showImgs, title):

    imgCopy = img.copy()
    imgBlur = cv2.GaussianBlur(imgCopy, (5, 5), 0)
    cannyImg = cv2.Canny(imgBlur, 0, 50)
    cannyImg = cv2.dilate(cannyImg, np.ones((2, 2), np.uint8), iterations=2)

    if showImgs:
        cv2.imshow("CannyResized", resizeImg(cannyImg, imgScale=0.5))

    lines = cv2.HoughLinesP(cannyImg, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=int(img.shape[1]*1/16.53), maxLineGap= int(img.shape[1]*0.004));

    # Method Variables
    rotatedImg = []
    angleFound = False
    sumSkewAngles = 0
    maxlen = int(imgCopy.shape[1] / 3)  # Only consider lines with a substantial length (i.e. quarter of the image width)
    counter = 1

    for i in range(lines.shape[0]):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        if abs(x2 - x1) > maxlen:  # Only Horizontal Lines considered
            angleDiff = -math.atan2((y2 - y1), (x2 - x1))

            if abs(angleDiff) > 0.001 and abs(angleDiff) < .08 and (abs(angleDiff) > sumSkewAngles / counter or counter < 3):
                sumSkewAngles = sumSkewAngles + angleDiff
                counter = counter + 1
                angleFound = True
                cv2.line(imgCopy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if angleFound:
        rotatedImg = imutils.rotate_bound(img, sumSkewAngles/counter * 180 / np.pi)

    if showImgs:
        cv2.imshow(title, resizeImg(imgCopy, imgScale = 0.5))
        if len(rotatedImg):
            cv2.imshow(title + "rotated", resizeImg(rotatedImg, imgScale = 0.5))
        cv2.waitKey()
        cv2.destroyAllWindows()

    if angleFound:
        return rotatedImg

def checkVerticalSkew(img, showImgs, title):

    imgCopy = img.copy()
    imgBlur = cv2.GaussianBlur(imgCopy, (5, 5), 0)
    cannyImg = cv2.Canny(imgBlur, 0, 50)
    cannyImg = cv2.dilate(cannyImg, np.ones((2, 2), np.uint8), iterations=4)

    if showImgs:
        cv2.imshow("CannyResized", resizeImg(cannyImg, imgScale=0.5))


    # Method Variables
    lines = cv2.HoughLinesP(cannyImg, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=int(img.shape[0] * 1 / 23.39), maxLineGap=int(img.shape[0] / 233.9));
    rotatedImg = []
    angleFound = False
    sumSkewAngles = 0
    maxlen = int(imgCopy.shape[0] / 4)  # Only consider lines with a substantial length (i.e. quarter of the image height)
    counter = 1

    for i in range(lines.shape[0]):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        #cv2.line(imgCopy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if abs(y2 - y1) > maxlen:  # Only Vertical Lines considered
            angleDiff = math.atan2((y2 - y1), (x2 - x1))  #- np.pi / 2
            if angleDiff > 0:
                angleDiff = np.pi / 2 - angleDiff
            else:
                angleDiff = abs(angleDiff) - np.pi / 2

            if abs(angleDiff) >= 0 and abs(angleDiff) < .06 and (abs(angleDiff) > sumSkewAngles/counter or counter < 3):
                sumSkewAngles = angleDiff + sumSkewAngles
                counter = counter + 1
                angleFound = True
                cv2.line(imgCopy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if angleFound:
        rotatedImg = imutils.rotate_bound(img, (sumSkewAngles/counter) * (180 / np.pi))


    if showImgs:
        cv2.imshow(title, resizeImg(imgCopy, imgScale = 0.5))
        if len(rotatedImg):
            cv2.imshow(title + "rotated", resizeImg(rotatedImg, imgScale = 0.5))
        cv2.waitKey()
        cv2.destroyAllWindows()

    if angleFound:
        return rotatedImg
    else:
        return []

def debugVertical(debugPath, debug):
    if debug:
        checkVerticalSkew(cv2.imread(debugPath), showImgs=True, title="debugVertical")

def debugHorizontal(debugPath, debug):
    if debug:
        checkHorizontalSkew(cv2.imread(debugPath), showImgs=True, title="debugHorizontal")

def checkSkewness(img, path):

    showImgs = False
    debugVertical(debugPath = "/home/greggas/Desktop/PDF_imgs/debug2.jpg", debug = False)
    debugHorizontal(debugPath="/home/greggas/Desktop/PDF_imgs/debug2.jpg", debug = False)

    # Check Vertical Lines:
    rotatedImg = checkVerticalSkew(img, showImgs=showImgs, title = "Vertical Lines")
    if len(rotatedImg):
        cv2.imwrite(path, rotatedImg)

    # Check Horizontal Lines if no Vertical Found
    else:
        rotatedImg = checkHorizontalSkew(img, showImgs=showImgs, title = "Horizontal Lines")
        if rotatedImg is not None:
            cv2.imwrite(path, rotatedImg)
        else:
            cv2.imwrite(path, img) # Write original file to disk when no skewness was found

    #if showImgs:

'''________________________________________________________________________________'''


def testfunc():

    #mask = swtObj.scrub('/home/greggas/Desktop/testImg.png')
    #cv2.imshow("mask", resizeImg(mask, 0.5))
    #cv2.imwrite("mask.jpg", mask)
    #cv2.waitKey()

    img = cv2.imread('/home/greggas/Desktop/testImg.png', 0)
    imgCopy = img.copy()
    cv2.bitwise_not(img, imgCopy);


    kernel = np.ones((2, 2), np.uint8)

    imgCopy = cv2.morphologyEx(imgCopy, cv2.MORPH_OPEN, kernel)
    imgCopy = cv2.morphologyEx(imgCopy, cv2.MORPH_CLOSE, kernel)
    imgCopy = cv2.morphologyEx(imgCopy, cv2.MORPH_OPEN, kernel)
    imgCopy = cv2.morphologyEx(imgCopy, cv2.MORPH_CLOSE, kernel)
    imgCopy = cv2.morphologyEx(imgCopy, cv2.MORPH_OPEN, kernel)
    imgCopy = cv2.morphologyEx(imgCopy, cv2.MORPH_CLOSE, kernel)

    #cv2.bitwise_not(imgCopy, imgCopy);
    cv2.imwrite("imgCopy.jpg", imgCopy)



    cv2.imshow("imgCopy", resizeImg(imgCopy, 0.5))

    #cv2.imwrite("mask.jpg", mask)

    #cv2.waitKey()

    swtObj = SWTScrubber()
    mask = swtObj.scrub('/home/greggas/Desktop/testImg.png')

    #mask = cv2.imread("mask.jpg", 0)

    ret, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
    cv2.imwrite("mask.jpg", mask)
    cv2.imshow("mask", resizeImg(mask, 0.5))
    cv2.imshow("imgCopy", resizeImg(imgCopy, 0.5))
    print("type(mask) : ", type(mask))
    print("type(imgCopy) : ", type(imgCopy))

    print("mask.shape, : ", mask.shape)
    cv2.imwrite("mask.jpg", mask)
    print("imgCopy.shape : ", imgCopy.shape)
    cv2.waitKey()

    ret, imgCopy = cv2.threshold(imgCopy, 5, 255, cv2.THRESH_BINARY)

    #imgAnd = cv2.bitwise_and(imgCopy, mask)#, mask=mask_inv)
    #cv2.imshow("imgAnd", resizeImg(imgAnd, 0.5))

    #cv2.bitwise_not(imgAnd, imgAnd);
    #cv2.imshow("imgAndNot", resizeImg(imgAnd, 0.5))

    cv2.imshow("mask", resizeImg(mask, 0.5))
    cv2.imwrite("mask.jpg", mask)
    cv2.waitKey()

    #edges, sobelx64f, sobely64f, theta = swtObj._create_derivative('/home/greggas/Desktop/testImg.png')
    #cv2.imshow("edges", resizeImg(edges, 0.5))
    #cv2.imshow("sobelx64f", resizeImg(sobelx64f, 0.5))
    #cv2.imshow("sobely64f", resizeImg(sobely64f, 0.5))
    #cv2.imshow("testImg", resizeImg(img, 0.5))


    #swt = swtObj._swt(theta, edges, sobelx64f, sobely64f)
    #cv2.imshow("swt", resizeImg(swt, 0.5))
    #cv2.waitKey()

    img = cv2.imread('/home/greggas/Desktop/noisyImg.png', 0)
    cv2.imshow("testImg", resizeImg(img, 0.5))
    #cv2.waitKey()
    imgCopy = img.copy()

    cv2.bitwise_not(img, imgCopy);
    cv2.imshow("bitwise_not", resizeImg(imgCopy, 0.5))
    #cv2.waitKey()

    dist_transform = cv2.distanceTransform(imgCopy, cv2.DIST_L2, 3)
    cv2.imshow("dist_transform", resizeImg(dist_transform, 0.5))
    cv2.waitKey()

    #imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # get the bounding rect
        area = cv2.contourArea(c)
        if area < 5:
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(img, [c], 0, (255, 255, 255), 3)
            # draw a white rectangle to visualize the bounding rect
            cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (255, 255, 255), 5)
            #cv2.imshow("imgCopy", resizeImg(img.copy(), imgScale=0.5))

    cv2.imshow("imgCopy", resizeImg(imgCopy, imgScale=0.5))
    cv2.waitKey()
    cv2.destroyAllWindows()


def testfunc2():
    if True:

        #img = cv2.imread('/home/greggas/Desktop/PDF_imgs/04-20002-0100_73168-0.jpg', 0)
        img = cv2.imread('imgCopy.jpg', 0)

        imgCopy = img.copy()
        y = 0
        x = 0
        winH = int(imgCopy.shape[0] / 2)
        winW = int(imgCopy.shape[1] / 2)
        crop_img = imgCopy[y:y + winH, x:x + winW]
        ret, crop_img = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("cropped", crop_img)
        output = image_to_string(crop_img, lang='eng')
        print(output)

        cv2.waitKey()

        imgCopy = img.copy()
        # imgBlur = cv2.GaussianBlur(imgCopy, (5, 5), 0)
        cannyImg = cv2.Canny(imgCopy, 0, 50)
        # cannyImg = cv2.dilate(cannyImg, np.ones((1, 1), np.uint8), iterations=4)

        if True:
            cv2.imshow("CannyResized", resizeImg(cannyImg, imgScale=0.5))

        # Method Variables
        lines = cv2.HoughLinesP(cannyImg, rho=1, theta=1 * np.pi / 180, threshold=100,
                                minLineLength=int(img.shape[0] * 1 / 23.39), maxLineGap=int(img.shape[0] / 233.9));

        maxlen = int(imgCopy.shape[0] * imgCopy.shape[
            0] / 64)  # Only consider lines with a substantial length (i.e. quarter of the image height)
        counter = 1
        cv2.bitwise_not(img, imgCopy);

        for i in range(lines.shape[0]):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]

            lenLine = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1)
            # cv2.line(imgCopy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if lenLine > maxlen:
                cv2.line(imgCopy, (x1, y1), (x2, y2), (0, 0, 0), 2)

        cv2.imshow("HoughLines", resizeImg(imgCopy, imgScale=0.5))
        cv2.waitKey()
        # ret, cleanedImg = \
        # cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)  #
        # cv2.bitwise_not(img, imgCopy);

        # ret, cleanedImg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow("imgCopy", imgCopy)
        # cv2.waitKey()

        # contours, hierarchy = cv2.findContours(cleanedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hier = cv2.findContours(imgCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            # get the bounding rect
            if cv2.contourArea(c) > 200:
                x, y, w, h = cv2.boundingRect(c)
                # draw a white rectangle to visualize the bounding rect
                cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (255, 0, 0), 5)
                cv2.imshow("imgCopy", resizeImg(imgCopy, imgScale=0.5))
                cv2.waitKey()

        cv2.destroyAllWindows()
    # cv2.drawContours(imgCopy, contours, -1, (255, 0, 0), 5)
    # cv2.imshow("example", resizeImg(imgCopy, imgScale=0.5))
    # cv2.waitKey()
    # cv2.imwrite("output.png", img)


def findAllRect(img):

   # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((2, 2), np.uint8)
    #img = cv2.dilate(img, kernel, iterations=2)


    # print("outputOld = ", outputOld)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


    imgCopy = img.copy()
    ret, imgCopy = cv2.threshold(imgCopy, 80, 255, 1)
    imgBlur = cv2.GaussianBlur(imgCopy, (5, 5), 0)
    cannyImg = cv2.Canny(imgBlur, 0, 240)
    kernel = np.ones((2, 2), np.uint8)
    imgCopy = cv2.dilate(imgCopy, kernel, iterations=2)
    imgCopy = cv2.dilate(imgCopy, kernel, iterations=2)
    imgCopy = cv2.dilate(imgCopy, kernel, iterations=2)

    cv2.imshow('imgBlur', resizeImg(imgBlur, 0.5))
    cv2.imshow('Canny', resizeImg(cannyImg, 0.5))
    cv2.imshow('imgCopy', resizeImg(imgCopy, 0.5))
    cv2.imshow('Canny', resizeImg(cannyImg, 0.5))
    cv2.waitKey()

    imgCopy = img.copy()
    lines = cv2.HoughLinesP(cannyImg, rho=1, theta=1 * np.pi / 180, threshold=20, minLineLength=int(img.shape[1]*1/30), maxLineGap= int(img.shape[1]*0.0025));

    # Method Variables
    angleFound = False
    maxSkewAngle = 0
    maxlen = int(imgCopy.shape[1] / 25)  # Only consider lines with a substantial length (i.e. quarter of the image height)
    N = lines.shape[0]

    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        lenLine = math.sqrt((y2 - y1)*(y2 - y1) + (x2 - y1)*(x2 - y1))

        if lenLine > maxlen:  # Only Vertical Lines considered

            cv2.line(imgCopy, (x1, y1), (x2, y2), (0, 0, 0), 2)


    cv2.imshow('imgCopy', resizeImg(imgCopy, 0.5))
    cv2.waitKey()
    ret, thresh = cv2.threshold(imgCopy, 127, 255, 1)

    contours, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )


    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    imgCopy = img.copy()
    for cnt in contours:
        #imgCopy = img.copy()
        drawn = False

        epsVals = 0

        for i in range(0, 40):
            epsVals = epsVals + 0.01
            approx = cv2.approxPolyDP(cnt, epsVals * cv2.arcLength(cnt, False), True)
            if len(approx) == 4:
                if cv2.contourArea(cnt) > 2000:
                    topLeft, bottomRight = findTopLBottomR(approx)
                    area = (bottomRight[1] - topLeft[1]) * (bottomRight[0] - topLeft[0])
                    if area > 2000:
                        cv2.rectangle(imgCopy, topLeft, bottomRight, (0, 255, 0), 3)
                        drawn = True

            if drawn:
                cv2.imshow('imgCopy', resizeImg(imgCopy, 0.5))
                cv2.waitKey()
                break


    cv2.imshow('imgCopy', resizeImg(imgCopy, 0.5))
    cv2.waitKey()
    cv2.destroyAllWindows()



def findTopLBottomR(approx):

    arrX = [approx[0][0][0], approx[1][0][0], approx[2][0][0], approx[3][0][0]]
    arrY = [approx[0][0][1], approx[1][0][1], approx[2][0][1], approx[3][0][1]]
   # FIND TOPLEFT -------------------------------
    m1, m2 = float('inf'), float('inf')
    for x in arrX:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x

    ix1 = arrX.index(m1)
    ix2 = arrX.index(m2)
    y1 = arrY[ix1]
    y2 = arrY[ix2]

    if y1 < y2:
        x1 = arrX[ix1]
    else:
        x1 = arrX[ix2]
        y1 = y2

    topLeft = (x1, y1)

   # FIND BottomRight -------------------------------

    m1, m2 = 0, 0
    for x in arrX:
        if x >= m1:
            m1, m2 = x, m1
        elif x > m2:
            m2 = x

    ix1 = arrX.index(m1)
    ix2 = arrX.index(m2)
    y1 = arrY[ix1]
    y2 = arrY[ix2]
    if y1 > y2:
        x1 = arrX[ix1]
    else:
        x1 = arrX[ix2]
        y1 = y2


    return topLeft, (x1, y1)









'''
   OLD CODE:
   _-------------------------------------------------------------------------------------------
   def checkSkewness2(img, path):

    # Check Vertical Lines:
    rotatedImg = []
    
    if 1:
        #debugVertical(debugPath = "/home/greggas/Desktop/PDF_imgs/debug2.jpg", debug = True)
        rotatedImg = checkVerticalSkew(img, showImgs=True, title = "Vertical Lines")

    #rotatedImg = []
    if len(rotatedImg):
        cv2.imwrite(path, rotatedImg)

    # Check Horizontal Lines if no Vertical Found
    else:
        #debugHorizontal(debugPath="/home/greggas/Desktop/PDF_imgs/debug2.jpg", debug=True)
        rotatedImg = checkHorizontalSkew(img, showImgs=True, title = "Horizontal Lines")

    cv2.waitKey()
    
    _-------------------------------------------------------------------------------------------
    
    
    
   def checkHorizontalSkew(img, showImgs, title): # TODO - NEED TO FINISH
    d = True
    imgCopy = img.copy()
    imgBlur = cv2.GaussianBlur(imgCopy, (5, 5), 0)
    cannyImg = cv2.Canny(imgBlur, 10, 50)
    lines = cv2.HoughLinesP(cannyImg, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50);

    # Method Variables
    angleFound = False
    maxSkewAngle = 0
    maxlen = int(imgCopy.shape[1] / 4)  # Only consider lines with a substantial length (i.e. quarter of the image width)
    N = lines.shape[0]

    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        if abs(x2 - x1) > maxlen:  # Only Horizontal Lines considered
            angleDiff = math.atan2((y2 - y1), (x2 - x1))
            #if tempAngle < 0:
             #   angleDiff = tempAngle
            #angleDiff = tempAngle
            #angleDiff = tempAngle
            #angleDiff = abs(tempAngle) - np.pi
            #print(angleDiff)
            # TODO: use Average Angle instead of max in case one angle is a lemon/rogue/outlier one
            if abs(angleDiff) > 0.015 and abs(angleDiff) < .06 and abs(angleDiff) > abs(maxSkewAngle):

                maxSkewAngle = -angleDiff
                
               # if angleDiff > 0:
                #    maxSkewAngle = -angleDiff
                #else:
                    maxSkewAngle = - angleDiff  # NEED TO ROTATE BACKWARDS!!
                
                print(maxSkewAngle)
                angleFound = True
                cv2.line(imgCopy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if angleFound:
        rotatedImg = imutils.rotate_bound(img, maxSkewAngle * 180 / np.pi)

    if showImgs:
        cv2.imshow(title, resizeImg(imgCopy, imgScale=0.5))
        cv2.imshow(title + "rotated", resizeImg(rotatedImg, imgScale=0.5))

    if angleFound:
        return rotatedImg
        
        
   def checkVerticalSkew(img, showImgs, title):

    imgCopy = img.copy()
    imgBlur = cv2.GaussianBlur(imgCopy, (5, 5), 0)
    cannyImg = cv2.Canny(imgBlur, 10, 50)
    lines = cv2.HoughLinesP(cannyImg, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50);

    # Method Variables
    angleFound = False
    maxSkewAngle = 0
    maxlen = int(imgCopy.shape[0] / 4)  # Only consider lines with a substantial length (i.e. quarter of the image height)
    N = lines.shape[0]

    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        if abs(y2 - y1) > maxlen:  # Only Vertical Lines considered
            tempAngle = math.atan2((y2 - y1), (x2 - x1))
            angleDiff = abs(tempAngle) - np.pi / 2

            # TODO: use Average Angle instead of max in case one angle is a lemon/rogue/outlier one
            if abs(angleDiff) > 0.015 and abs(angleDiff) < .06 and abs(angleDiff) > abs(maxSkewAngle):

                if tempAngle > 0:
                    maxSkewAngle = -angleDiff
                else:
                    maxSkewAngle = angleDiff  # NEED TO ROTATE BACKWARDS!!

                print(maxSkewAngle)
                angleFound = True
                cv2.line(imgCopy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if angleFound:
        rotatedImg = imutils.rotate_bound(img, maxSkewAngle * 180 / np.pi)


    if showImgs:
        cv2.imshow(title, resizeImg(imgCopy, imgScale = 0.5))
        cv2.imshow(title + "rotated", resizeImg(rotatedImg, imgScale = 0.5))

    if angleFound:
        return rotatedImg
    else:
        return []

    
   
   '''



# TODO:
''' if Orientation is skewed but not extreme like 90/180/270 degrees
Try Implementation: https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033

'''

#!/usr/bin/python
from PDF_Reader.PDF_Reader_Functions import sliding_window,resizeImg, improveImgQuality
import numpy as np
from pytesseract import image_to_string
import threading
import time
import cv2
import re


exitFlag = 0

class PDFThread (threading.Thread):
    def __init__(self, threadID, name, dilateKernel, img, dataToSearch):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.threadName = name
        self.kernel = dilateKernel
        self.imgFile = img
        self.terms = dataToSearch
        self.data = []
        self.notFoundData = []
        self.progress = 10

    def run(self):
        print(self.name, " started:")
        self.cleanImgCollectData(self.imgFile, self.kernel, self.terms, self.threadName)

    def cleanImgs(self, imgFiles, kernel):
        # Clean and Enhance PDF page images
        return cv2.imread(imgFiles)

        if 0:
            cleanedImgs = []

            for i, img in enumerate(imgFiles):
                cleanedImg = cv2.imread(img)
                # blur = cv2.blur(img, (5, 5))
                # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                # ret, cleanedImg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                # cleanedImg = cv2.dilate(cleanedImg, kernel, iterations=2)
                cleanedImgs.append(cleanedImg)
                print(self.name, "has appended image(s)")

            return cleanedImgs

    def collectData(self, imgs, dataToSearch):
        img = imgs

        (winW, winH) = (int(img.shape[1]*.35), int(img.shape[0]*.04))
        stepSize = int(img.shape[1]*0.03)
        #print(stepSize)
        # loop over the image pyramid
        termFound = False
        counter = 0
        tempCounter = 0
        for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(winW, winH)):
            progressCalc = int(100*y/img.shape[0])
            if progressCalc > self.progress:
                print(self.threadName, " has scanned ", self.progress, "% of the document '", self.imgFile, "'")
                self.progress += 10
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            if termFound == True and tempCounter == counter:
                x = x - stepSize


            dataList, termFound, counter, notFoundList = self.collectInLoop(img, [x, y, winH, winW, dataToSearch], termFound, counter)

            if termFound == True and tempCounter + 2 < counter:
                tempCounter = counter


            if dataList is not None:
                self.data.append(dataList)

            if self.notFoundData is not None:
                self.notFoundData.append(notFoundList)



            # print("lock released")

        if 0:
            for i, img in enumerate(imgs):
                (winW, winH) = (int(img.shape[1] * .75), int(img.shape[0] * .03))
                #if i < 2:
                    #continue
                # loop over the image pyramid

                for (x, y, window) in sliding_window(img, stepSize=50, windowSize=(winW, winH)):

                    if window.shape[0] != winH or window.shape[1] != winW:
                        continue

                    threadLock.acquire()
                    #print("lock acquired")
                    reading = self.collectInLoop(img, [x,y,winH,winW,searchedTerms])
                    if reading is not None:
                        self.data.append(reading)

                    threadLock.release()

                    #print("lock released")

    def collectInLoop(self, img, params, termFound, counter):

        dataList = None
        notFoundList = None
        x, y, winH, winW, searchedTerms = params[0], params[1], params[2], params[3], params[4],

        # since we do not have a classifier, we'll just draw the window
        crop_img = img.copy()[y:y + winH, x:x + winW]
        ret, crop_img = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY)

        try:
            meanPixelVal = np.mean(crop_img, axis=(0, 1))[0]
        except:
            IndexError
            meanPixelVal = 250

        # IGNORE BLACK IMAGES (If an image was rotated, could have been black in parts -ie corners)
        if meanPixelVal < 247 and meanPixelVal > 180:

            #print("crop_img.shape: ", crop_img.shape)
            #cv2.waitKey(100)
            # print(self.name, " output = ", output[0:10] + "...")
            if termFound or counter % 3 == 0:
                output = image_to_string(resizeImg(crop_img, 1.5), lang='eng')
                #if self.imgFile == "/home/greggas/Desktop/PDF_imgs/collections-1.jpg":
                    #print("output = ", output)
                tempNum = "0"
                try:
                    tempNum = re.sub("\D", "", output)#str(int(filter(str.isdigit, output)))
                except:
                    TypeError
                #print(tempNum)
                termFound = False
                if len(tempNum) > 3:
                    #print(self.name, " thread with output: ", output)
                    for j, term in enumerate(searchedTerms):
                        if term in output:
                            threadLock = threading.Lock()
                            threadLock.acquire()
                            print(self.name," found the term '", term, "' in image-file '", self.imgFile, "'.")
                            threadLock.release()
                            dataList = improveImgQuality(crop_img, output, term, searchedTerms)
                            termFound = True

                if len(tempNum) == 0:
                    for j, term in enumerate(searchedTerms):
                        if term in output:
                            threadLock = threading.Lock()
                            threadLock.acquire()
                            print(self.name," found the term '", term, "' in image-file '", self.imgFile, "'.")
                            print("No numbers found: Will check with different window at later stage")
                            threadLock.release()
                            notFoundList = [x, y, self.imgFile, term]

        counter = counter + 1

        return dataList, termFound, counter, notFoundList


    def cleanImgCollectData(self, imgFile, dilateKernel, dataToSearch, threadName):
        cleanedImgs = self.cleanImgs(imgFile, dilateKernel)
        #print(type(cleanedImgs))
        #cv2.imshow(self.threadName + " Image", cleanedImgs)
        cv2.imwrite('/home/greggas/Desktop/Thread_Files/file_' + threadName +'.png',cleanedImgs)
        #cv2.waitKey()
        #print("type(cleanedImgs) = ", type(cleanedImgs))
        print(self.name, "started analysing page")
        self.collectData(cleanedImgs, dataToSearch)

    def print_time(threadName, counter, delay):
       while counter:
          if exitFlag:
             threadName.exit()
          time.sleep(delay)
          print("%s: %s" % (threadName, time.ctime(time.time())))
          counter -= 1


'''____________________Create Threads____________________'''

def createThreads(imgFiles, dilateKernel, dataToSearch):
    listThreads = []
    listNotFoundTerms = []
    savedData = open("savedData.txt","w")

    for i in range(0, len(imgFiles)):
        # Create threads depending on number of imgs
        threadName = "Thread-" + str(i+1)

        t = PDFThread(threadID=i, name=threadName, dilateKernel=dilateKernel, img=imgFiles[i],
                      dataToSearch=dataToSearch)
        try:
            t.start()
            listThreads.append(t)
        except:
            print("Error: unable to start thread")

    # Wait for all threads to complete
    for t in listThreads:
        t.join()
        if len(t.data): # Check list is not empty
            for j in range(0, len(t.data)):
                for k in range(0, len(t.data[j])):
                    t.data[j][k] = (t.data[j][k]).replace("\n", "<<<<<<") # remove new lines in string and replace with "<<<<<<" # Avoid numbers on new line from being appended
                    savedData.write(t.data[j][k] + "\n")
        if len(t.notFoundData):
            listNotFoundTerms.append(t.notFoundData)

    savedData.close()

    print("Threading finished")
    return listNotFoundTerms



'''____________________Filtering Information____________________'''


def extractInfo(dataToSearch, txtfile):#, listNotFoundTerms):    # Needs to be at least 8 numbers, no letters:
                                           # Spaces have to be removed
    termsFound = []
    savedData = open(txtfile, "r")
    if savedData.mode == "r":
        savedData = savedData.readlines()

    for i in range(0, len(savedData)):
        item = savedData[i]

        for j, term in enumerate(dataToSearch):
            numStr = ""
            if term in item:
                counter = 0
                loc = item.find(term)
                termEntrySet = False
                for k in range(loc+len(term), len(item)):
                    val = item[k]

                    if not termEntrySet:
                        termEntry = term
                        if termEntry == "HS" or termEntry == "H S":
                            termEntry = "H.S"

                        if termEntry == "Tarif" or termEntry == "Harmonised" or termEntry == "tarif" or termEntry == "Harmonized" or termEntry == "Harmonized classification":
                            termEntry = "Harmonised Tarif"

                        if termEntry == "douane":
                            termEntry = "Douane"

                        termEntrySet = True

                    if val == " " or val == ":" or val == ";" or val == "-" or val == ".":
                        continue

                    if val == "<" or val == "/": # Ensure numbers on newlines are not appended and Dates are not mistaken as numbers
                        counter += 10

                    if val.isdigit():
                        numStr += val

                    if val.isalpha() and len(numStr) > 1:
                        counter += 15

                    elif counter > 30 and len(numStr) > 5:  # Prevent other characters besides numbers being added to string

                        termsFound.append([termEntry, numStr])
                        break

                    elif counter > 30:
                        break

                    else:
                        counter += 1

                if len(numStr) > 7:
                    termsFound.append([termEntry, numStr])

    #filteredDict = buildDict(termsFound)
    return buildDict(termsFound)
    '''
    for j in range(0, len(listNotFoundTerms)):
       x, y, imgFile, term = listNotFoundTerms[i]
       if term in filteredDict:
           continue

       else:
            img = cv2.imread(imgFile)
            # Check with different Window for the term:
            (winW, winH) = (int(img.shape[1] * .3), int(img.shape[0] * .2))

            # loop over the image pyramid

            termFound = False
            counter = 0
            #for (x, y, window) in sliding_window(img, stepSize=50, windowSize=(winW, winH)):

               # if the window does not meet our desired window size, ignore it
               #if window.shape[0] != winH or window.shape[1] != winW:
               #continue
               # startTime = time.time()
               # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
               # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
               # WINDOW

            # since we do not have a classifier, we'll just draw the window
            clone = img.copy()
            crop_img = clone[y:y + winH, x:x + winW]
            ret, crop_img = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY)
            # kernel = np.ones((1, 1), np.uint8)
            # crop_img = cv2.dilate(crop_img, kernel, iterations=2)
            # mean = np.mean(crop_img.all())
            mean = np.mean(crop_img, axis=(0, 1))[0]
            # print(mean)
            # mean, stddev = cv2.meanStdDev(cv2.UMat(crop_img.all()))

            # cv2.waitKey(1000)
            # IGNORE BLACK IMAGES
            if mean < 247 and mean > 180:

               # temp = resizeImg(clone, 40)#scalePercent=40)
               # cv2.imshow("clone", clone)
               # cv2.waitKey(100)
               # clone = resizeImg(clone, scalePercent=40)
               # crop_img = resizeImg(crop_img, 1.5)

               if termFound or counter % 3 == 0:
                   crop_img = resizeImg(crop_img, 1.5)
                   output = image_to_string(crop_img, lang='eng')
                   # print(output)
                   # cv2.waitKey(50)
                   termFound = False
                   cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                   cv2.imshow("cropped", crop_img)
                   cv2.imshow("Window", resizeImg(clone, imgScale=0.5))
                   cv2.waitKey(50)
                   tempNum = "0"
                   try:
                       tempNum = re.sub("\D", "", output)  # str(int(filter(str.isdigit, output)))
                   except:
                       TypeError
                   termFound = False
                   if len(tempNum) > 2:
                       for j, term in enumerate(dataToSearch):
                           if term in output:
                               # print("Term was Found! Term = ", term, ", output = ", output)
                               output = improveImgQuality(crop_img, output, term, dataToSearch)
                               data.append(output)
                               termFound = True
                       # cv2.waitKey()

               # if wordToSearch in output:
               # print("Searched string found!! Output = ", output, "wordToSearch = ", wordToSearch)
               # endTime = time.time()
               # print("Time taken = ", endTime - startTime)
               counter = counter + 1

               # time.sleep(0.00001)
    '''




def buildDict(termsFound):

    # Choose Most Likely Candidates:
    dictCandidates = {}

    for i in range(0, len(termsFound)):

        term = termsFound[i][0]
        id = termsFound[i][1]

        if term not in dictCandidates:
            dictCandidates[term] = []
            dictCandidates[term].append([id, 1])

        else:
            for j in range(0, len(dictCandidates[term])):
                if dictCandidates[term][j][0] == id:
                    dictCandidates[term][j][1] += 1
                    break
                elif j == len(dictCandidates[term]) - 1:
                    dictCandidates[term].append([id, 1])

    return filterDict(dictCandidates)

def filterDict(dictCandidates):

    # Final Dictionary Created Based on Maximum Number of Hits:
    finalDict = {}
    print(dictCandidates)
    for key in dictCandidates:
        maxHits = 1
        for i in range(0, len(dictCandidates[key])):
            #lenID = len(dictCandidates[key][i][0])

            if dictCandidates[key][i][1] > maxHits:# and lenID > maxLenID:
                maxHits = dictCandidates[key][i][1]
                #maxLenID = lenID
                if maxHits > 3:
                    finalDict[key] = dictCandidates[key][i][0]

    # Remove Duplicate Terms
    removeCodeKey = False
    for key in finalDict:
        if key == "Code":
            val = finalDict[key]
            for key2 in finalDict:
                if key2 != "Code":
                    val2 = finalDict[key2]
                    if val2 == val:
                        removeCodeKey = True

    if removeCodeKey:
        finalDict.pop("Code", None)

    return finalDict

#def


def returnImgListInfo(imgFiles):

    lenImgList = len(imgFiles)
    numCycles = 0
    remCycles = lenImgList % 10 - 1

    if lenImgList > 10:  # Run Max 10 Threads at once:
        numCycles = int(lenImgList / 10)

    return numCycles, remCycles
































'''
#num = re.findall(term, item)
try:
num = int(''.join(filter(str.isdigit, item)))
except ValueError:
print("invalid literal for int() with base 10")

#indexTerm = item.find(term)
#indexNum = item.find(str(num))
#print("term = ", term, " at index ", indexTerm)
print("num = ", num)#, " at index ", indexNum)
print("item = ", item)
#print(indexNum, indexTerm)

# Validate Number:
numStr = str(num)
startIndexOfNum = 0
prevChar = ""
for k in range(0, len(item)):
currChar = item[k]
if currChar == item[startIndexOfNum]:
startIndexOfNum = k
else:
startIndexOfNum += 1







numStr = str(num)
numStrInTerm = ""
notSolved = True
subsetNotSolved = True
k = 0
numStrInTerm = numStr[k]
counter2 = 0


while k < len(numStr) - 1 and counter2 < 20:
counter = 0
num_k_addOne = numStr[k+1]
print("numStr = ", numStr)
print("item = ", item)
print("num_k_addOne = ", num_k_addOne)
numStrInTermTemp = numStrInTerm + num_k_addOne
print("numStrInTermTemp = ", numStrInTermTemp)
indexK = item.find(numStrInTermTemp)#re.findall(numStrInTermTemp, item)
print("indexK = ", indexK)
ktemp = k

while indexK == -1 and counter < 3:
print(numStrInTermTemp)
numStrInTermTemp = numStrInTerm + " "
indexK =  item.find(numStrInTermTemp)#re.findall(numStrInTermTemp, item)
counter += 1
print(numStrInTermTemp, " with length: ", len(numStrInTermTemp), "\n")
k = ktemp-1

numStrInTerm = numStrInTermTemp

k += 1
print("k = ", k)
counter2 += 1
print(numStrInTerm)
print("k = ", k)
'''
'''
num_k = numStr[k]
indexK = re.findall(num_k, term)
if indexK != -1:

if type(num) == int:
termsFound.append([term, num])
print(termsFound)
'''
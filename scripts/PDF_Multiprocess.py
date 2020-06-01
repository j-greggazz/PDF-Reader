from PDF_Reader.PDF_Reader_Functions import sliding_window, resizeImg, improveImgQuality
import numpy as np
from pytesseract import image_to_string
import multiprocessing as mp
import time
import cv2
import re
from os import getpid

l = mp.Lock()
class pdfProcess:
    def __init__(self, multipID, name, dilateKernel, img, dataToSearch):
        self.processID = multipID
        self.processName = name
        self.kernel = dilateKernel
        self.imgFile = img
        self.terms = dataToSearch
        self.data = []
        self.notFoundData = []
        self.progress = 10

def collectData(img, pdfProcess):
    (winW, winH) = (int(img.shape[1] * .35), int(img.shape[0] * .04))
    stepSize = int(img.shape[1] * 0.03)
    # print(stepSize)
    # loop over the image pyramid
    termFound = False
    counter = 0
    tempCounter = 0
    for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(winW, winH)):
        progressCalc = int(100 * y / img.shape[0])
        if progressCalc > pdfProcess.progress:
            print(pdfProcess.processName, " has scanned ", pdfProcess.progress, "% of the document '", pdfProcess.imgFile, "'")
            pdfProcess.progress += 10
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        if termFound == True and tempCounter == counter:
            x = x - int(stepSize/2)

        # print("lock acquired")
        dataList, termFound, counter, notFoundList = collectInLoop(pdfProcess, img, [x, y, winH, winW, pdfProcess.terms],
                                                                        termFound, counter)

        if termFound == True and tempCounter + 2 < counter:
            tempCounter = counter

        if dataList is not None:
            pdfProcess.data.append(dataList)


        # print("lock released")
    return pdfProcess.data
    #writeToFile(pdfProcess)

def collectInLoop(pdfProcess, img, params, termFound, counter):

    dataList = None
    notFoundList = None
    x, y, winH, winW, searchedTerms = params[0], params[1], params[2], params[3], params[4],

    crop_img = img.copy()[y:y + winH, x:x + winW]
    ret, crop_img = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY)

    try:
        meanPixelVal = np.mean(crop_img, axis=(0, 1))[0]
    except:
        IndexError
        meanPixelVal = 250

    if meanPixelVal < 247 and meanPixelVal > 180:

        if termFound or counter % 3 == 0:
            output = image_to_string(resizeImg(crop_img, 1.5), lang='eng')

            tempNum = "0"
            try:
                tempNum = re.sub("\D", "", output)  # str(int(filter(str.isdigit, output)))
            except:
                TypeError

            termFound = False
            if len(tempNum) > 3:

                for j, term in enumerate(searchedTerms):
                    if term in output:
                        l.acquire()
                        print(pdfProcess.processName, " found the term '", term, "' in image-file '", pdfProcess.imgFile, "'.")
                        l.release()
                        dataList = improveImgQuality(crop_img, output, term, searchedTerms)
                        termFound = True

            if len(tempNum) == 0:
                for j, term in enumerate(searchedTerms):
                    if term in output:
                        l.acquire()
                        print(pdfProcess.processName, " found the term '", term, "' in image-file '", pdfProcess.imgFile, "'.")
                        print("No numbers found: Will check with different window at later stage")
                        l.release()
                        notFoundList = [x, y, pdfProcess.imgFile, term]

    counter = counter + 1

    return dataList, termFound, counter, notFoundList

def cleanImgs(imgFiles, kernel):
    # Clean and Enhance PDF page images
    return cv2.imread(imgFiles)

def cleanImgCollectData(pdfProcess):#imgFile, dilateKernel, dataToSearch, processName):
    cleanedImg = cleanImgs(pdfProcess.imgFile, pdfProcess.kernel)
    cv2.imwrite('/home/greggas/Desktop/Process_Files/file_' + pdfProcess.processName + '.png', cleanedImg)
    print(pdfProcess.processName, "started analysing page")
    return collectData(cleanedImg, pdfProcess)

def writeToFile(data):

    if len(data):  # Check list is not empty
        savedData = open("savedData.txt", "a+")
        for j in range(0, len(data)):
            for k in range(0, len(data[j])):
                data[j][k] = (data[j][k]).replace("\n","<<<<<<")  # remove new lines in string and replace with "<<<<<<" # Avoid numbers on new line from being appended
                savedData.write(data[j][k] + "\n")
                savedData.flush()
        savedData.close()

'''____________________Create Processes____________________'''


def run(pdfProcess):
    print(pdfProcess.processName, " started:")
    return cleanImgCollectData(pdfProcess)


def createProcesses(num_processes, imgFiles, dilateKernel, dataToSearch):
    listProcesses = []
    listNotFoundTerms = []
    #savedData = open("savedData.txt", "w")

    # pool = mp.Pool(num_processes)
    # processName = "Process-" + str(i + 1)
    pool = mp.Pool(num_processes)
    # pdfProcess = PDF_MultiProcess(multipID=i, name=processName, dilateKernel=dilateKernel, img=imgFiles[i],
    # dataToSearch=dataToSearch)
    processes = []
    queue = mp.Queue()
    for i in range(num_processes):
        p = pdfProcess(multipID = i, name = "Process-" + str(i + 1), dilateKernel = dilateKernel , img = imgFiles[i], dataToSearch = dataToSearch)
        processes.append(pool.apply_async(run, args=(p,)))

    #processes = [pool.apply_async(run, args=(["Process-" + str(i + 1), dilateKernel, imgFiles[i]], dataToSearch)) for i in range(num_processes)]
    listData = []
    for p in processes:
        result = p.get()
        listData.append(result)

    pool.close()
    pool.join()

    #for i in range(0, len(listData)):
        #print(listData[i])
    for i in range(0, len(listData)):
        writeToFile(listData[i])

    '''
    for i in range(num_processes):
        processName = "Process-" + str(i + 1)
        pdfProcess = PDF_MultiProcess(multipID=i, name=processName, dilateKernel=dilateKernel, img=imgFiles[i],
                                      dataToSearch=dataToSearch)
        p = pool.apply_async(run(pdfProcess))
        try:
            p.start()
            listProcesses.append(p)
        except:
            print("Error: unable to start process")
    
    for i in range(0, len(imgFiles)):

        # Create Processes depending on number of imgs

        processName = "Process-" + str(i + 1)
        pdfProcess = PDF_MultiProcess(multipID = i, name = processName, dilateKernel = dilateKernel, img=imgFiles[i], dataToSearch = dataToSearch)
        p = mp.Process(target = run(pdfProcess))
        p.get()


        try:
            p.start()
            listProcesses.append(p)
        except:
            print("Error: unable to start process")
    
    # Wait for all Processes to complete
    for p in listProcesses:
        p.join()
        if len(p.data):  # Check list is not empty
            for j in range(0, len(p.data)):
                for k in range(0, len(p.data[j])):
                    p.data[j][k] = (p.data[j][k]).replace("\n",
                                                          "<<<<<<")  # remove new lines in string and replace with "<<<<<<" # Avoid numbers on new line from being appended
                    savedData.write(p.data[j][k] + "\n")
        if len(p.notFoundData):
            listNotFoundTerms.append(p.notFoundData)

    savedData.close()
    '''
    print("Processing finished")
    return listNotFoundTerms

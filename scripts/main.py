import numpy as np
import time
import cv2
#import pyopencl
from PDF_Reader.PDF_Reader_Functions import convertPDFtoImgs, cleanImgCollectData, resizeImg, testfunc, testfunc2, findAllRect
from PDF_Reader.PDFThread import createThreads, extractInfo, returnImgListInfo
from PDF_Reader.PDF_Multiprocess import createProcesses
import multiprocessing as mp
from pytesseract import image_to_string, image_to_osd, TesseractError
#from pyopencl.tools import get_test_platforms_and_devices

# Variables
datafiles = ["collections"] #["9087A0003-02_F1958"] #["98N7830H022-000_F3700"] ["04-20002-0100_73168"] # ["1209-100_F1976_extract"] ["27A01-30006-21_F6137"] ["98N7830H022-000_F3700"]
save_dir = '../data/output/'
load_dir = '../data/input/'
txtFile = "savedData.txt"
dilateKernel = np.ones((1, 1), np.uint8)
dataToSearch = ["HTS", "Harmonized Tarif", "Harmonised Tarif", "Harmonized classification", "Customs", "Zolltarif",
                "Classification", "douane", "Douane", "Commodity", "Tarif", "tarif", "Harmonised", "Harmonized", "H.S", "H S", "HS", "Code", "Douanier"]

readFromTxtDoc = False
multithreading = False
multiprocessing = True

if __name__ == '__main__':

    #img = cv2.imread('../data/input/04-20002-0100_73168-0.jpg', 0)
    #findAllRect(img)
    #testfunc()
    #testfunc2()
    # print(get_test_platforms_and_devices())


    if readFromTxtDoc:
        finalDict = extractInfo(dataToSearch, txtFile)
        print("finalDict = ", finalDict)

    else:
        start = time.time()
        for j in range(0, len(datafiles)):

            ''' 1. RETURN PDF TO EDIT '''
            filenamePDF  = load_dir + datafiles[j] + ".pdf"


            ''' 2. CONVERT PDF TO IMAGE '''
            imgFiles = convertPDFtoImgs(filenamePDF, save_dir)


        ''' 3. CLEAN IMAGES AND COLLECT DATA '''

        if multithreading:
            numCycles, remCycles = returnImgListInfo(imgFiles)

            for i in range(0, numCycles):
                #print("10 * i = ", 10*i)
                #print("10 * i + 9 = ", 10 * i + 9)
                createThreads(imgFiles[10 * i: 10 * i + 9], dilateKernel, dataToSearch)

            #print("numCycles*10 = ", numCycles*10)
            #print("numCycles * 10 + remCycles = ",  numCycles * 10 + remCycles)

            listNotFoundTerms = createThreads(imgFiles[numCycles*10: numCycles*10+remCycles], dilateKernel, dataToSearch)
            finalDict = extractInfo(dataToSearch, txtFile) #, listNotFoundTerms)
            print("finalDict = ", finalDict)

        elif multiprocessing:
            num_processes = mp.cpu_count()
            #rem = len(imgFiles) % num_processes
            counter = 1
            #rem = len(arr) % num_processes
            for i in range(0, len(imgFiles), num_processes):
                rangeFiles = imgFiles[(counter - 1) * num_processes:counter * num_processes]
                createProcesses(len(rangeFiles), rangeFiles, dilateKernel, dataToSearch)
                counter += 1

            finalDict = extractInfo(dataToSearch, txtFile)
            print("finalDict = ", finalDict)

        else:
            cleanImgCollectData(imgFiles, dilateKernel, dataToSearch)


        end = time.time()
        #"{0:.2f}".format(a)
print("Time-taken = ", "{0:.2f}".format((end - start)/60), " mins.")
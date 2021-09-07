import os
import sys
import numpy as np
import cv2 as cv
import pyautogui
import win32gui, win32ui, win32con, win32api
from time import time, sleep
from datetime import datetime  
import dlib
from pynput.keyboard import Key, Controller

# Zoom View Classification
SOLO = "solo.png"
MULTIPLE = "multiple.png"
BIG = "big.png"

SPEAKER = "speaker.png"
ONE = "one.png"
TWO = "two.png"
THREE = "three.png"
FOUR = "four.png"
FIVE = "five.png"
SIX = "six.png"
SEVEN = "seven.png"
EIGHT = "eight.png"
NINE = "nine.png"

sample_names = [
    SPEAKER,
    ONE,
    TWO,
    THREE,
    FOUR,
    SOLO,
    BIG,
]
#    FIVE,
#    SIX,
#    SEVEN,
#    EIGHT,
#    NINE,
#    MULTIPLE,
    

#sample_names = [SPEAKER, TWO]

# Read image sample files
sample_images = []
for img_filename in sample_names:
    sample_images.append(cv.imread(img_filename))

# Convert screenshot to grayscale
sample_grays = []
for image in sample_images:
    sample_grays.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

# Threshold the image    
sample_threses = []
for gray in sample_grays:
    _, thresh = cv.threshold(gray, 30, 255, cv.THRESH_BINARY) # 30 is ok
    sample_threses.append(thresh)
    
# Get View Classification Histogram getting Eucidean Distance
histogram_list = []
for thresh in sample_threses:
    histogram = cv.calcHist([thresh], [0], None, [256], [0, 256])
    histogram_list.append(histogram)


def delay(sec):
    if sec > 0:
        print("In", sec, end='')
        sys.stdout.flush()
        for i in range(sec):
            print(" .", end='')
            sys.stdout.flush()
            sleep(1)
        print("")


def get_screenshot():
    myScreenshot = pyautogui.screenshot()
    #myScreenshot.save("screenshot.png")

    # Convert pyautogui to opencv
    open_cv_image = np.array(myScreenshot) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    return open_cv_image
    

def list_window_names():
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), '"' + win32gui.GetWindowText(hwnd) + '"')
    win32gui.EnumWindows(winEnumHandler, None)

        
def get_window_names():
    names = []
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            #print(hex(hwnd), '"' + win32gui.GetWindowText(hwnd) + '"')
            names.append(win32gui.GetWindowText(hwnd))
    win32gui.EnumWindows(winEnumHandler, None)
    return names
    
def get_zoom_view_classification(screenshot):
    global histogram_list
    
    
    # https://www.geeksforgeeks.org/measure-similarity-between-images-using-python-opencv/
    
    # test image
    #image = cv2.imread('cat.jpg')
    gray_image = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray_image, 30, 255, cv.THRESH_BINARY) # 30 is ok 
    input_histogram = cv.calcHist([threshed], [0], None, [256], [0, 256])
    
    # Initialize list of zeros to handle the euclidean distances
    euclidean_distance_list = [0] * len(histogram_list)
    
    #c1, c2 = 0, 0
    
    # Compute every Euclidean distance for every Zoom Sample
    for index in range(len(histogram_list)):
        # Euclidean Distance between screenshot and sample
        i = 0
        while i<len(input_histogram) and i<len(histogram_list[index]):
            euclidean_distance_list[index] \
                +=(input_histogram[i]-histogram_list[index][i])**2
            i+= 1
        euclidean_distance_list[index] = euclidean_distance_list[index]**(1 / 2)
      
    #print("euclidean_distance_list", euclidean_distance_list)
    
    
    # Return the index (which means the number of videos visible)
    # with the smallest value.
      
    min_value = min(euclidean_distance_list)
    min_index = euclidean_distance_list.index(min_value)
    
    return min_index
        # 0 is one, 1 is two, 8 is nine
        # 9 is big, 10 is solo, 11 is multiple, 12 is speaker

def match_templating_zoom_screen(screenshot):
    # https://www.youtube.com/watch?v=ffRYijPR8pk

    global sample_threses

    # Convert screenshot to grayscale
    gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    # after that, we doing thresholding on image
    #_, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
    _, thresh = cv.threshold(gray, 30, 255, cv.THRESH_BINARY) # 30 is ok for gallery view of two or more
    
    #cv.imwrite("screenshot.png", thresh)

    thresholded_grayed_screenshot = thresh
    
    match_results = []
    
    for sample in sample_threses:
        #results = cv.matchTemplate(sample, thresholded_grayed_screenshot, cv.TM_SQDIFF_NORMED)
        results = cv.matchTemplate(sample, thresholded_grayed_screenshot, cv.TM_CCOEFF_NORMED)
        
        # Get the minimum from the results as the best match
        minimum = max(results)
        match_results.append(minimum) ## minimum is a list of one element
        
    # Return the max as the best match
    max_value = max(match_results)
    max_index = match_results.index(max_value)
    
    #print("match_results:", match_results)
    #print("max_index:", max_index)

    return max_index
        # 0 is one, 1 is two, 8 is nine
        # 9 is big, 10 is solo, 11 is multiple, 12 is speaker


    haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_UNCHANGED)
    needle_img = cv.imread('albion_cabbage.jpg', cv.IMREAD_UNCHANGED)
    result = cv.matchTemplate(haystack_img, needle_img, cv.TM_SQDIFF_NORMED)

    threshold = 0.17
    # The np.where() return value will look like this:
    # (array([482, 483, 483, 483, 484], dtype=int32), array([514, 513, 514, 515, 514], dtype=int32))
    locations = np.where(result <= threshold)
    # We can zip those up into a list of (x, y) position tuples
    locations = list(zip(*locations[::-1]))
    #print(locations)


def main():
    obs_window_name = None
    zoom_meeting_window_name = None
    
    for name in get_window_names():
        #print(name)
        if "OBS" in name:
            obs_window_name = name
            break
            
    for name in get_window_names():
        #print(name)
        if "OBS" in name:
            zoom_meeting_window_name = name
            break
    
    # If OBS window and Zoom Meeting window is found...
    if obs_window_name and zoom_meeting_window_name:
    
        detector = dlib.get_frontal_face_detector()
        loop_time = time()
        
        last_face_count = 0
        is_zoom_meeting_active = False
        is_waiting_for_active_zoom_meeting_message_already_printed = False
        no_change_printed_once = False
        
        
        obs_window_handle = win32gui.FindWindow(None, obs_window_name)
        zoom_window_handle = win32gui.FindWindow(None, zoom_meeting_window_name)
        keyboard = Controller()
        
        while(True):
            dt_object = datetime.fromtimestamp(time())
            if not is_zoom_meeting_active:
                if not is_waiting_for_active_zoom_meeting_message_already_printed:
                    print("Waiting for Zoom Meeting to be the active window...")
                    is_waiting_for_active_zoom_meeting_message_already_printed = True
        
            # Get currently active window name
            active_window_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            
            # If Zoom Meeting is the active window, meaning infront of the screen...
            #if not "Zoom Meeting" in active_window_name: # Not the optimal condition
                # because "Chrome - Charles Jayson Dadios's Zoom Meeting"
                # still contains the string "Zoom Meeting"
            # If the active window name doesn't start with "Zoom Meeting"
            if not active_window_name.find("Zoom Meeting") == 0: # Use this insted
                is_zoom_meeting_active = False
                no_change_printed_once = False # reset no. 1
            else:
                # If the active window name starts with "Zoom Meeting"
                
                is_zoom_meeting_active = True
                is_waiting_for_active_zoom_meeting_message_already_printed = False
                
                # Capture frame-by-frame
                #delay(3)
                
                # Count the visible faces...assume one face per window
                image = get_screenshot()
                cv.imwrite("screenshot.png", image)
                
                #euclidean_classification = get_zoom_view_classification(image)
                templating_classification = match_templating_zoom_screen(image)
                
                #print("euclidean_classification:", euclidean_classification)
                #print("templating_classification:", templating_classification)

                
                # The number represent a classification of Zoom screen.
                face_count = templating_classification # Templating classification does better 
                #print("Active layout:", sample_names[face_count])
                
                
                #print("last_face_count:", last_face_count)
                #print("current face_count:", face_count)

                
                if face_count == last_face_count:
                    if no_change_printed_once == False:
                        print("No change...")
                        no_change_printed_once = True
                else:
                    # The face count changed...
                    print("Detected faces:", face_count)
                    #print("obs_window_name:", obs_window_name)
                    
                    dt_object = datetime.fromtimestamp(time())
                    print("timestamp:", dt_object)
                       
                    if True: # Control Zoom
                        # Set OBS to foreground to send keystrokes to it.
                        #delay(1)
                        win32gui.SetForegroundWindow(obs_window_handle)
                        
                        # Press a key depending on the Zoom Meeting window
                        
                        if sample_names[face_count] == SOLO:
                            pyautogui.press("s") # Screensharing with regular sized solo speaker
                        elif sample_names[face_count] == MULTIPLE:
                            pyautogui.press("s") # Screensharing with regular sized multiple speaker
                        elif sample_names[face_count] == BIG:
                            pyautogui.typewrite("sss") # Screensharing with big sized solo speaker
                        elif sample_names[face_count] == SPEAKER:
                            pyautogui.press("f") # Speaker view, the biggest solo screen
                        elif sample_names[face_count] == ONE:
                            pyautogui.press("1")
                        elif sample_names[face_count] == TWO:
                            pyautogui.press("2")
                        elif sample_names[face_count] == THREE:
                            pyautogui.press("3")
                        elif sample_names[face_count] == FOUR:
                            pyautogui.press("4")
                        elif sample_names[face_count] == FIVE:
                            pyautogui.press("5")
                        elif sample_names[face_count] == SIX:
                            pyautogui.press("6")
                        elif sample_names[face_count] == SEVEN:
                            pyautogui.press("7")
                        elif sample_names[face_count] == EIGHT:
                            pyautogui.press("8")
                        elif sample_names[face_count] == NINE:
                            pyautogui.press("9")
                            
                        
                        #pyautogui.press("down")
                        
                        # Set back Zoom Meeting to foreground to count open cameras.
                        # Press Alt + Tab
                        
                        # Automatically return to Zoom from OBS
                        #delay(2)
                        #pyautogui.keyDown("alt") # Holds down the alt key
                        #pyautogui.press("tab") # Presses the tab key once
                        #pyautogui.keyUp("alt") # Lets go of the alt key
         
                        #win32gui.SetForegroundWindow(zoom_window_handle) # not working
                    # End if control Zoom
                    else: # Do nothing in OBS, but just log on the terminal
                        print("Match layout:", sample_names[face_count])
                    
                    # Update last_face_count
                    last_face_count = face_count
                    no_change_printed_once = False # reset no. 2
                
                #break
            # End else do something about OBS
            
            
            
            
            #cv.imshow("Screenshot",gray)
            #cv.imwrite("screenshot.png", gray)
            #cv.imwrite("screenshot.png", image_enhanced)
            
            # debug the loop rate
            #print('FPS {}'.format(1 / (time() - loop_time)))
            #loop_time = time()
            
            
            # press 'q' with the output window focused to exit.
            # waits 1 ms every loop to process key presses
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                break
            
        # End while not pressing Ctrl + C   

    else:
        print("OBS not found.")
    

if __name__ == "__main__":
    main()
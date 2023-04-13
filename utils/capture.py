import numpy as np
import win32gui, win32ui, win32con
import cv2
import keyboard
import tensorflow as tf
import pyautogui
import mouse
#output format of MobileNet model = ['a', 'd', 'down', 'j', 'l', 'left', 'right', 's', 'space', 'up', 'w']
pyautogui.FAILSAFE = False
class WindowCapture:

    # properties
    w = 2560
    h = 1440
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name):
        # find the handle for the window we want to capture
        self.last_position_x = 0
        self.last_position_y = 0
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = 640
        self.h = 480

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img


    def record_with_inputs(self):
        def on_move( x, y):

            if x > self.last_position_x:  # [LEFT, RIGHT]
                X = [('left',False), ('right',True)]
            elif x < self.last_position_x:
                X = [('left',True), ('right',False)]
            else:
                X = [('left',False), ('right',False)]
            if y > self.last_position_y:  # [UP, DOWN]
                Y = [('up',True), ('down',False)]
            elif y < self.last_position_y:
                Y = [('up',False), ('down',True)]
            else:
                Y = [('up',False), ('down',False)]
            self.last_position_x = x
            self.last_position_y = y
            return X + Y
        k = 0
        print('works')
        while True:
            x, y = pyautogui.position()
            lr_ud = on_move(x,y)
            c = self.get_screenshot()
            cv2.imshow('Halo', c)
            key = cv2.waitKey(1)
            interested_inputs = ['w','s','a','d','e','space','j','l','f'] # plus left, right, up, and down controls
            lis = [(i, keyboard.is_pressed(i)) for i in interested_inputs]+lr_ud
            print(lis)

            for i,j in lis:
                if j == True:
                 cv2.imwrite(f'C:/Users/Dell/Desktop/Halo inputs/{i}/{k}.jpeg', c)
                 k += 1
            print(k)

    def return_prediction(self, drift = 7):
        model_path = 'C:/Users/Dell/PycharmProjects/MasterCheeks/CNN/MobileNet1.h5'
        MobileNet_model = tf.keras.models.load_model(model_path)
        x,y = 0,0
        check = ['a', 'd', 'down', 'j', 'l', 'left', 'right', 's', 'space', 'up', 'w']
        mouse_inp = {'down': mouse.drag(0, 0, 0, y, absolute=False, duration=0.1),'left': mouse.drag(100, 0, (-1)*x, 0, absolute=False, duration=0.1),'right': mouse.drag(0, 0, x, 0, absolute=False, duration=0.1) ,'up': mouse.drag(0, 100, 0, (-1)*y, absolute=False, duration=0.1)}
        while True:
            c = self.get_screenshot()
            im = cv2.resize(c,dsize = (480, 640), interpolation = cv2.INTER_CUBIC)
            # cv2.imshow('Halo', c)
            # key = cv2.waitKey(1)
            preds = MobileNet_model.predict(np.expand_dims(np.asarray(im),axis =0))
            move = check[np.argmax(preds)]
            print(move)
            keyboard.press(check[np.argmax(preds)])
            if move in mouse_inp.keys():
               mouse_inp[move]
            x+=drift
            y+=drift

    # def find_ammo(self, img):
        
    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    def list_window_names(self):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)

import time
import io
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.ttk import Label
import tkinter.filedialog
import PIL
import cv2
from keras.optimizers import Adam
from PIL import Image, ImageDraw, ImageTk, ImageFont

from controllers.ApplicationController import ApplicationController
from dao.UserDAO import UserDAO
from eye_correction.EyeFixer import EyeFixer
from eye_detection.EyeDetector import EyeDetector
from face_detection.FaceDetector import FaceDetector
from face_detection.Model import Model
from services.UserService import UserService

PATH_TO_SAVE_IMAGES = 'saved_images/'

class MyConfig:
    def __init__(self, frame_name, window_size, frame_controller, main_controller):
        self.__frame_name = frame_name
        self.__window_size = window_size
        self.__frame_controller = frame_controller
        self.__main_controller = main_controller

    def getFrameName(self):
        return self.__frame_name

    def getWindowSize(self):
        return self.__window_size

    def getFrameController(self):
        return self.__frame_controller

    def getMainController(self):
        return self.__main_controller

class RedEyeGUI(tk.Tk):
    def __init__(self, *args, **kwargs):
      tk.Tk.__init__(self, *args, **kwargs)
      self.container = tk.Frame(self)

      self.container.pack(side="top", fill="both", expand = True)

      self.container.grid_rowconfigure(0, weight=1)
      self.container.grid_columnconfigure(0, weight=1)

      self.__frames = {}

    def show_frame(self, page):
        frame = self.__frames[page]
        self.winfo_toplevel().title(frame.getFrameName())
        self.winfo_toplevel().geometry(frame.getWindowSize())
        frame.tkraise()

    def set_controller(self, controller_helper):
        self.__controller_helper = controller_helper
        self.add_frames()

    def add_frames(self):
        for F in (RegisterFrame, MainFrame, LoginFrame):
            frame = F(self.container, self, self.__controller_helper)
            self.__frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(LoginFrame)

class RegisterFrame(tk.Frame, MyConfig):
    def __init__(self, parent, controller, controller_helper):
        tk.Frame.__init__(self, parent)
        MyConfig.__init__(self, "Register", "300x250", controller, controller_helper)

        self.__username = tk.StringVar()
        self.__password = tk.StringVar()

        ttk.Label(self, text="Login").pack()
        ttk.Label(self, text="").pack()

        userLabel = ttk.Label(self, text="Username")
        userLabel.pack()
        self.userEntry = ttk.Entry(self, textvariable=self.__username)
        self.userEntry.pack()

        passwordLabel = ttk.Label(self, text="Password")
        passwordLabel.pack()

        self.passwordEntry = ttk.Entry(self, show="*", textvariable=self.__password)
        self.passwordEntry.pack()

        ttk.Label(self, text="").pack()

        registerButton = ttk.Button(self, text="Register", width=10,
                                   command=self.register)
        registerButton.pack()

    def register(self):
        print("Hello from register handler")
        if self.getMainController().register(self.__username.get(), self.__password.get()):
            messagebox.showinfo("Information", "The user has been created successfully")
            self.getFrameController().show_frame(LoginFrame)
        else:
            self.userEntry.delete(0, 'end')
            self.passwordEntry.delete(0, 'end')
            messagebox.showerror("Error", "Error")

class MainFrame(tk.Frame, MyConfig):
    def __init__(self, parent, controller, controller_helper):
        tk.Frame.__init__(self, parent)
        MyConfig.__init__(self, "Red Eye Removal Application", "800x600", controller, controller_helper)

        self.__image_path = None
        self.__clean_image = None
        self.__path_to_clean_image = None
        # tk.Tk.config(self, menu=actionmenu)

        self.__canvas = tk.Canvas(self, width = 800, height = 600)
        self.__canvas.grid(row=0, column=0)

        self.__img = ImageTk.PhotoImage(Image.open("gui/bg.png").resize((800, 600), Image.ANTIALIAS))
        self.__canvas.background = self.__img  # Keep a reference in case this code is put in a function.
        self.__bg = self.__canvas.create_image(0, 0, anchor=tk.NW, image=self.__img)
        # Put a tkinter widget on the canvas.
        button = tk.Button(self, text="Upload image", command=self.upload_action)
        self.__button_window = self.__canvas.create_window(10, 10, anchor=tk.NW, window=button)
        button.place(x = 0, y = 0)
        fix_button = tk.Button(self, text="Fix image", command=self.fix_image)
        self.__fix_button_window = self.__canvas.create_window(20, 20, anchor=tk.NW, window=fix_button)
        fix_button.place(x = 100, y = 0)

        save_button = tk.Button(self, text="Save image", command=self.save_image)
        self.__save_button_window = self.__canvas.create_window(20, 20, anchor=tk.NW, window=save_button)
        save_button.place(x = 180, y=0)

        # loginLabel = ttk.Label(self, text="Hello from MainFrame", justify=tk.CENTER)
        # loginLabel.pack()

    def save_image(self):
        if self.__path_to_clean_image != None:
            files = [('All Files', '*.*'),
                     ('PNG Files', '*.png'),
                     ('JPG Files', '*.jpg')]
            file = filedialog.asksaveasfile(filetypes = files, mode='w', defaultextension=".png")
            im = Image.open(self.__path_to_clean_image)
            im.save(file)
        else:
            messagebox.showerror("Error", "No image loaded")

    def fix_image(self):
        if self.__image_path != None:
            self.__clean_image = self.getMainController().fix_image(cv2.imread(self.__image_path, cv2.IMREAD_COLOR))
            cv2.imshow("image", self.__clean_image)
            self.__path_to_clean_image = PATH_TO_SAVE_IMAGES + 'im_' + str(int(time.time())) + "." + self.__image_path.split(".")[1]
            cv2.imwrite(self.__path_to_clean_image, self.__clean_image)
            self.change_bg_image(self.__path_to_clean_image)
        else:
            messagebox.showerror("Error", "No image loaded")

    def change_bg_image(self, path):
        self.__canvas.delete(self.__bg)
        photo = ImageTk.PhotoImage(Image.open(path).resize((800, 600), Image.ANTIALIAS))
        self.__bg = self.__canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.__canvas.background = photo  # prevent gbc to delete the image

    def upload_action(self, event=None):

        filename = filedialog.askopenfilename()
        self.change_bg_image(filename)
        self.__image_path = filename
        print('Selected:', filename)

class LoginFrame(tk.Frame, MyConfig):
    def __init__(self, parent, controller, controller_helper):
        tk.Frame.__init__(self, parent)
        MyConfig.__init__(self, "Login", "300x250", controller, controller_helper)

        self.__username = tk.StringVar()
        self.__password = tk.StringVar()

        ttk.Label(self, text="Login").pack()
        ttk.Label(self, text="").pack()

        userLabel = ttk.Label(self, text="Username")
        userLabel.pack()
        self.userEntry = ttk.Entry(self, textvariable=self.__username)
        self.userEntry.pack()

        passwordLabel = ttk.Label(self, text="Password")
        passwordLabel.pack()

        self.passwordEntry = ttk.Entry(self, show="*", textvariable=self.__password)
        self.passwordEntry.pack()

        ttk.Label(self, text="").pack()

        loginButton = ttk.Button(self, text="Login", width=10,
                                    command=self.login)

        loginButton.pack()

        goToRegisterButton = ttk.Button(self, text="Register", width=10,
                                    command=lambda: controller.show_frame(RegisterFrame))
        goToRegisterButton.pack()

    def login(self):
        print(self.__username.get(), self.__password.get())
        if self.getMainController().login(self.__username.get(), self.__password.get()):
            self.getFrameController().show_frame(MainFrame)
        else:
            self.userEntry.delete(0, 'end')
            self.passwordEntry.delete(0, 'end')
            messagebox.showerror("Error", "Incorrect Credentials")


def build_controller():
    user_dao = UserDAO()
    user_service = UserService(user_dao)

    input_shape = (28, 28, 3)
    modelBuilder = Model(input_shape, construct_model=True)
    lr = 1e-3
    epochs = 250
    loss_function = 'binary_crossentropy'
    optimizer = Adam(lr=lr, decay=lr / epochs)
    metrics = ['accuracy']
    loaded_model = modelBuilder.load_model_from_file()
    loaded_model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    face_detector = FaceDetector(loaded_model, 1.5, (128, 128), 32, 0.4)
    eye_detector = EyeDetector()
    eye_fixer = EyeFixer(face_detector, eye_detector)

    appController = ApplicationController(user_service, eye_fixer)
    return appController


controller = build_controller()
print("Hello From Here")
app = RedEyeGUI()
app.set_controller(controller)

app.mainloop()

# tk = tk.Tk()
# img = Image.open("gui/bg.png")
# img_bg = ImageTk.PhotoImage(img)
# background_label = Label(image=img_bg)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)
#
# tk.mainloop()
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import cv2
import csv
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import pymysql
import threading

# Create directories if they don't exist
if not os.path.exists('Manually Attendance'):
    os.makedirs('Manually Attendance')

if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

if not os.path.exists('StudentDetails'):
    os.makedirs('StudentDetails')

if not os.path.exists('TrainingImage'):
    os.makedirs('TrainingImage')

if not os.path.exists('TrainingImageLabel'):
    os.makedirs('TrainingImageLabel')

window = tk.Tk()
window.title("FAMS-Face Recognition Based Attendance Management System")
window.geometry('1280x720')
window.configure(background='grey80')

attendance_file_name = ""  # Variable to store the latest attendance file name

def clear():
    txt.delete(0, 'end')

def clear1():
    txt2.delete(0, 'end')

def del_sc1():
    sc1.destroy()

def err_screen():
    global sc1
    sc1 = tk.Tk()
    sc1.geometry('300x100')
    sc1.title('Warning!!')
    sc1.configure(background='grey80')
    Label(sc1, text='Enrollment & Name required!!!', fg='black',
          bg='white', font=('times', 16)).pack()
    Button(sc1, text='OK', command=del_sc1, fg="black", bg="lawn green", width=9,
           height=1, activebackground="Red", font=('times', 15, ' bold ')).place(x=90, y=50)

def del_sc2():
    sc2.destroy()

def err_screen1():
    global sc2
    sc2 = tk.Tk()
    sc2.geometry('300x100')
    sc2.title('Warning!!')
    sc2.configure(background='grey80')
    Label(sc2, text='Please enter your subject name!!!', fg='black',
          bg='white', font=('times', 16)).pack()
    Button(sc2, text='OK', command=del_sc2, fg="black", bg="lawn green", width=9,
           height=1, activebackground="Red", font=('times', 15, ' bold ')).place(x=90, y=50)

def take_img():
    l1 = txt.get()
    l2 = txt2.get()
    if l1 == '' or l2 == '':
        err_screen()
    else:
        enrollment_exists = False
        try:
            with open('StudentDetails/StudentDetails.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    if row and l1 == row[0]:  # Ensure row is not empty
                        enrollment_exists = True
                        break
        except FileNotFoundError:
            pass
        
        if enrollment_exists:
            Notification.configure(
                text="Enrollment number already exists!", bg="Red", width=50, font=('times', 18, 'bold'))
            Notification.place(x=250, y=400)
            return

        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            Enrollment = txt.get()
            Name = txt2.get()
            sampleNum = 0
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum += 1
                    cv2.imwrite("TrainingImage/" + Name + "." + Enrollment + '.' + str(sampleNum) + ".jpg",
                                gray[y:y + h, x:x + w])
                    cv2.imshow('Frame', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif sampleNum > 70:
                    break
            cam.release()
            cv2.destroyAllWindows()
            ts = time.time()
            Date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            Time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            row = [Enrollment, Name, Date, Time]
            with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile, delimiter=',')
                writer.writerow(row)
            res = "Images Saved for Enrollment : " + Enrollment + " Name : " + Name
            Notification.configure(
                text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
            Notification.place(x=250, y=400)
        except Exception as e:
            print(e)
            Notification.configure(text=str(e), bg="Red", width=50, font=('times', 18, 'bold'))
            Notification.place(x=250, y=400)

def testVal(inStr, acttyp):
    if acttyp == '1':  # If the action is an insertion
        if not inStr.isdigit():
            return False
    return True

def fill_attendance_thread(tx, Notifica):
    global attendance_file_name  # Use the global variable
    sub = tx.get()
    if sub == '':
        err_screen1()
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("TrainingImageLabel/Trainner.yml")
    except:
        Notifica.configure(text="Model not found, Please train model", bg="red", fg="black", width=33,
                           font=('times', 15, 'bold'))
        Notifica.place(x=20, y=250)
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        Notifica.configure(text="Cannot open camera", bg="red", fg="black", width=33, font=('times', 15, 'bold'))
        Notifica.place(x=20, y=250)
        return

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    df = pd.read_csv("StudentDetails/StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Enrollment', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    future = time.time() + 20
    while True:
        ret, im = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 70:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Enrollment'] == Id]['Name'].values
                tt = str(Id) + "-" + aa[0]
                attendance.loc[len(attendance)] = [Id, aa[0], date, timeStamp]
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
                cv2.putText(im, str(tt), (x + h, y), font, 1, (255, 255, 0,), 4)
            else:
                Id = 'Unknown'
                tt = str(Id)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                cv2.putText(im, str(tt), (x + h, y), font, 1, (0, 25, 255), 4)
        if time.time() > future:
            break
        attendance = attendance.drop_duplicates(['Enrollment'], keep='first')
        cv2.imshow('Filling attendance..', im)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    attendance_file_name = "Attendance/" + sub + "_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance = attendance.drop_duplicates(['Enrollment'], keep='first')
    attendance.to_csv(attendance_file_name, index=False)

    try:
        connection = pymysql.connect(host='localhost', user='root', password='', db='Face_reco_fill')  # Ensure the database exists
        cursor = connection.cursor()
    except Exception as e:
        print("Database connection failed:", e)
        Notifica.configure(text=f"Database connection failed: {str(e)}", bg="red", width=50, font=('times', 18, 'bold'))
        Notifica.place(x=20, y=250)
        return

    DB_Table_name = sub + "_" + date.replace('-', '_') + "_Time_" + Hour + "_" + Minute + "_" + Second
    sql = """CREATE TABLE IF NOT EXISTS `{}` (
             ID INT NOT NULL AUTO_INCREMENT,
             ENROLLMENT VARCHAR(100) NOT NULL,
             NAME VARCHAR(50) NOT NULL,
             DATE VARCHAR(20) NOT NULL,
             TIME VARCHAR(20) NOT NULL,
             PRIMARY KEY (ID)
             );""".format(DB_Table_name)

    insert_data = "INSERT INTO `{}` (ENROLLMENT, NAME, DATE, TIME) VALUES (%s, %s, %s, %s)".format(DB_Table_name)

    try:
        cursor.execute(sql)
        for index, row in attendance.iterrows():
            cursor.execute(insert_data, (row['Enrollment'], row['Name'], row['Date'], row['Time']))
        connection.commit()
    except Exception as ex:
        print("Data insertion failed:", ex)

    Notifica.configure(text='Attendance filled Successfully', bg="Green", fg="white", width=33, font=('times', 15, 'bold'))
    Notifica.place(x=20, y=250)

    cam.release()
    cv2.destroyAllWindows()

def check_sheets():
    if attendance_file_name:
        # Open the attendance file in a new window
        root = tk.Tk()
        root.title("Attendance Sheet")
        root.geometry('880x470')
        root.configure(background='grey80')

        with open(attendance_file_name, newline="") as file:
            reader = csv.reader(file)
            r = 0
            for col in reader:
                c = 0
                for row in col:
                    label = tk.Label(root, width=10, height=1, fg="black", font=('times', 15, ' bold '),
                                     bg="white", text=row, relief=tk.RIDGE)
                    label.grid(row=r, column=c)
                    c += 1
                r += 1
        root.mainloop()
    else:
        messagebox.showwarning("Warning", "No attendance sheet available. Please fill attendance first.")

def subjectchoose():
    def Fillattendances():
        thread = threading.Thread(target=fill_attendance_thread, args=(tx, Notifica))
        thread.start()

    windo = tk.Tk()
    windo.title("Enter subject name...")
    windo.geometry('580x320')
    windo.configure(background='grey80')
    Notifica = tk.Label(windo, text="Attendance filled Successfully", bg="Green", fg="white", width=33,
                        height=2, font=('times', 15, 'bold'))

    attf = tk.Button(windo, text="Check Sheets", command=check_sheets, fg="white", bg="black",
                     width=12, height=1, activebackground="white", font=('times', 14, 'bold'))
    attf.place(x=430, y=255)

    sub = tk.Label(windo, text="Enter Subject : ", width=15, height=2,
                   fg="black", bg="grey", font=('times', 15, 'bold'))
    sub.place(x=30, y=100)

    tx = tk.Entry(windo, width=20, bg="white", fg="black", font=('times', 23))
    tx.place(x=250, y=105)

    fill_a = tk.Button(windo, text="Fill Attendance", fg="white", command=Fillattendances, bg="SkyBlue1", width=20, height=2,
                       activebackground="white", font=('times', 15, 'bold'))
    fill_a.place(x=250, y=160)
    windo.mainloop()

def admin_panel():
    win = tk.Tk()
    win.title("LogIn")
    win.geometry('880x420')
    win.configure(background='grey80')

    def log_in():
        username = un_entr.get()
        password = pw_entr.get()

        if username == 'admin' and password == 'admin123':
            win.destroy()
            import tkinter as tk
            root = tk.Tk()
            root.title("Student Details")
            root.configure(background='grey80')

            with open('StudentDetails/StudentDetails.csv', newline="") as file:
                reader = csv.reader(file)
                r = 0
                for col in reader:
                    c = 0
                    for row in col:
                        label = tk.Label(root, width=10, height=1, fg="black", font=('times', 15, ' bold '),
                                         bg="white", text=row, relief=tk.RIDGE)
                        label.grid(row=r, column=c)
                        c += 1
                    r += 1
            root.mainloop()
        else:
            valid = 'Incorrect ID or Password'
            Nt.configure(text=valid, bg="red", fg="white",
                         width=38, font=('times', 19, 'bold'))
            Nt.place(x=120, y=350)

    Nt = tk.Label(win, text="", bg="Green", fg="white", width=40,
                  height=2, font=('times', 19, 'bold'))

    un = tk.Label(win, text="Enter username : ", width=15, height=2, fg="black", bg="grey",
                  font=('times', 15, ' bold '))
    un.place(x=30, y=50)

    pw = tk.Label(win, text="Enter password : ", width=15, height=2, fg="black", bg="grey",
                  font=('times', 15, ' bold '))
    pw.place(x=30, y=150)

    un_entr = tk.Entry(win, width=20, bg="white", fg="black",
                       font=('times', 23))
    un_entr.place(x=290, y=55)

    pw_entr = tk.Entry(win, width=20, show="*", bg="white",
                       fg="black", font=('times', 23))
    pw_entr.place(x=290, y=155)

    c0 = tk.Button(win, text="Clear", command=lambda: un_entr.delete(0, 'end'), fg="white", bg="black", width=10, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
    c0.place(x=690, y=55)

    c1 = tk.Button(win, text="Clear", command=lambda: pw_entr.delete(0, 'end'), fg="white", bg="black", width=10, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
    c1.place(x=690, y=155)

    Login = tk.Button(win, text="LogIn", fg="black", bg="SkyBlue1", width=20,
                      height=2,
                      activebackground="Red", command=log_in, font=('times', 15, ' bold '))
    Login.place(x=290, y=250)
    win.mainloop()

def trainimg():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        faces, Ids = getImagesAndLabels("TrainingImage")
        if len(faces) == 0 or len(Ids) == 0:
            raise ValueError("No faces or IDs found. Ensure images are in the correct format and directory.")

        recognizer.train(faces, np.array(Ids))
        recognizer.save("TrainingImageLabel/Trainner.yml")

        Notification.configure(text="Model Trained", bg="olive drab", width=50, font=('times', 18, 'bold'))
        Notification.place(x=250, y=400)
    except Exception as e:
        Notification.configure(text=f"Training failed: {str(e)}", bg="Red", width=50, font=('times', 18, 'bold'))
        Notification.place(x=250, y=400)
        print(f"Training failed: {str(e)}")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faceSamples = []
    Ids = []
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Ensure the detector is available

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)
        except Exception as e:
            print(f"Error processing image {imagePath}: {str(e)}")

    return faceSamples, Ids

def manually_fill():
    sb = tk.Tk()
    sb.title("Enter subject name...")
    sb.geometry('580x320')
    sb.configure(background='grey80')

    def err_screen_for_subject():
        def ec_delete():
            ec.destroy()
        global ec
        ec = tk.Tk()
        ec.geometry('300x100')
        ec.title('Warning!!')
        ec.configure(background='snow')
        Label(ec, text='Please enter your subject name!!!', fg='red',
              bg='white', font=('times', 16, ' bold ')).pack()
        Button(ec, text='OK', command=ec_delete, fg="black", bg="lawn green", width=9, height=1, activebackground="Red",
               font=('times', 15, ' bold ')).place(x=90, y=50)

    def fill_attendance():
        subb = SUB_ENTRY.get()
        if subb == '':
            err_screen_for_subject()
        else:
            sb.destroy()
            MFW = tk.Tk()
            MFW.title("Manually attendance of " + str(subb))
            MFW.geometry('880x470')
            MFW.configure(background='grey80')

            def del_errsc2():
                errsc2.destroy()

            def err_screen1():
                global errsc2
                errsc2 = tk.Tk()
                errsc2.geometry('330x100')
                errsc2.title('Warning!!')
                errsc2.configure(background='grey80')
                Label(errsc2, text='Please enter Student & Enrollment!!!', fg='black', bg='white',
                      font=('times', 16, ' bold ')).pack()
                Button(errsc2, text='OK', command=del_errsc2, fg="black", bg="lawn green", width=9, height=1,
                       activebackground="Red", font=('times', 15, ' bold ')).place(x=90, y=50)

            def testVal(inStr, acttyp):
                if acttyp == '1':  # insert
                    if not inStr.isdigit():
                        return False
                return True

            ENR = tk.Label(MFW, text="Enter Enrollment", width=15, height=2, fg="black", bg="grey",
                           font=('times', 15))
            ENR.place(x=30, y=100)

            STU_NAME = tk.Label(MFW, text="Enter Student name", width=15, height=2, fg="black", bg="grey",
                                font=('times', 15))
            STU_NAME.place(x=30, y=200)

            ENR_ENTRY = tk.Entry(MFW, width=20, validate='key',
                                 bg="white", fg="black", font=('times', 23))
            ENR_ENTRY['validatecommand'] = (
                ENR_ENTRY.register(testVal), '%P', '%d')
            ENR_ENTRY.place(x=290, y=105)

            def remove_enr():
                ENR_ENTRY.delete(0, 'end')

            STUDENT_ENTRY = tk.Entry(
                MFW, width=20, bg="white", fg="black", font=('times', 23))
            STUDENT_ENTRY.place(x=290, y=205)

            def remove_student():
                STUDENT_ENTRY.delete(0, 'end')

            def enter_data_DB():
                ENROLLMENT = ENR_ENTRY.get()
                STUDENT = STUDENT_ENTRY.get()
                if ENROLLMENT == '' or STUDENT == '':
                    err_screen1()
                else:
                    ts = time.time()
                    Date = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    DB_table_name = f"{subb}_{Date}_manual"

                    # Create CSV file for manual attendance
                    csv_file_name = f"Manually Attendance/{subb}_{Date}.csv"
                    with open(csv_file_name, mode='a+', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])  # Header
                        csv_writer.writerow([ENROLLMENT, STUDENT, Date, timeStamp])  # Data

                    try:
                        connection = pymysql.connect(
                            host='localhost', user='root', password='', db='manually_fill_attendance')  # Ensure the database exists
                        cursor = connection.cursor()
                    except Exception as e:
                        print("Database connection failed:", e)
                        return

                    sql = f"""CREATE TABLE IF NOT EXISTS `{DB_table_name}` (
                             ID INT NOT NULL AUTO_INCREMENT,
                             ENROLLMENT VARCHAR(100) NOT NULL,
                             NAME VARCHAR(50) NOT NULL,
                             DATE VARCHAR(20) NOT NULL,
                             TIME VARCHAR(20) NOT NULL,
                             PRIMARY KEY (ID)
                             );"""

                    insert_data = f"INSERT INTO `{DB_table_name}` (ENROLLMENT, NAME, DATE, TIME) VALUES (%s, %s, %s, %s)"
                    VALUES = (ENROLLMENT, STUDENT, Date, timeStamp)

                    try:
                        cursor.execute(sql)
                        cursor.execute(insert_data, VALUES)
                        connection.commit()
                    except Exception as ex:
                        print("Data insertion failed:", ex)

                    ENR_ENTRY.delete(0, 'end')
                    STUDENT_ENTRY.delete(0, 'end')

            DATA_SUB = tk.Button(MFW, text="Enter Data", command=enter_data_DB, fg="black", bg="SkyBlue1", width=20,
                                 height=2, activebackground="white", font=('times', 15, ' bold '))
            DATA_SUB.place(x=170, y=300)

            c1ear_enroll = tk.Button(MFW, text="Clear", command=remove_enr, fg="white", bg="black", width=10,
                                     height=1, activebackground="white", font=('times', 15, ' bold '))
            c1ear_enroll.place(x=690, y=100)

            c1ear_student = tk.Button(MFW, text="Clear", command=remove_student, fg="white", bg="black", width=10,
                                      height=1, activebackground="white", font=('times', 15, ' bold '))
            c1ear_student.place(x=690, y=200)

            MFW.mainloop()

    SUB = tk.Label(sb, text="Enter Subject : ", width=15, height=2,
                   fg="black", bg="grey80", font=('times', 15, ' bold '))
    SUB.place(x=30, y=100)

    SUB_ENTRY = tk.Entry(sb, width=20, bg="white",
                         fg="black", font=('times', 23))
    SUB_ENTRY.place(x=250, y=105)

    fill_manual_attendance = tk.Button(sb, text="Fill Attendance", command=fill_attendance, fg="black", bg="SkyBlue1", width=20, height=2,
                                       activebackground="white", font=('times', 15, ' bold '))
    fill_manual_attendance.place(x=250, y=160)
    sb.mainloop()

# Main GUI setup continues
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System", bg="black", fg="white", width=50,
                   height=3, font=('times', 30, ' bold '))
message.place(x=80, y=20)

Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15,
                        height=3, font=('times', 17))

lbl = tk.Label(window, text="Enter Enrollment : ", width=20, height=2,
               fg="black", bg="grey", font=('times', 15, 'bold'))
lbl.place(x=200, y=200)

txt = tk.Entry(window, validate="key", width=20, bg="white",
               fg="black", font=('times', 25))
txt['validatecommand'] = (txt.register(testVal), '%P', '%d')
txt.place(x=550, y=210)

lbl2 = tk.Label(window, text="Enter Name : ", width=20, fg="black",
                bg="grey", height=2, font=('times', 15, ' bold '))
lbl2.place(x=200, y=300)

txt2 = tk.Entry(window, width=20, bg="white",
                fg="black", font=('times', 25))
txt2.place(x=550, y=310)

clearButton = tk.Button(window, text="Clear", command=clear, fg="white", bg="black",
                        width=10, height=1, activebackground="white", font=('times', 15, ' bold '))
clearButton.place(x=950, y=210)

clearButton1 = tk.Button(window, text="Clear", command=clear1, fg="white", bg="black",
                         width=10, height=1, activebackground="white", font=('times', 15, ' bold '))
clearButton1.place(x=950, y=310)

AP = tk.Button(window, text="Check Registered students", command=admin_panel, fg="black",
               bg="SkyBlue1", width=19, height=1, activebackground="white", font=('times', 15, ' bold '))
AP.place(x=990, y=410)

takeImg = tk.Button(window, text="Take Images", command=take_img, fg="black", bg="SkyBlue1",
                    width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=90, y=500)

trainImg = tk.Button(window, text="Train Images", fg="black", command=trainimg, bg="SkyBlue1",
                     width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=390, y=500)

FA = tk.Button(window, text="Automatic Attendance", fg="black", command=subjectchoose,
               bg="SkyBlue1", width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
FA.place(x=690, y=500)

quitWindow = tk.Button(window, text="Manually Fill Attendance", command=manually_fill, fg="black",
                       bg="SkyBlue1", width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=990, y=500)

window.mainloop()

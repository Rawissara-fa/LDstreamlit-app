import streamlit as st
import pandas as pd
# import pydeck as pdk
from urllib.error import URLError
import components.algorithm as algorithm

import cv2
import math
import numpy as np
from datetime import datetime
from PIL import Image

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth



st.set_page_config(page_title="Parameter", page_icon=" ", layout="wide")
st.title("Parameter")
st.markdown("_________________________________________________")


# Check authentication when user lands on the page.
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)
    
authenticator = stauth.Authenticate(
config['credentials'],
config['cookie']['name'],
config['cookie']['key'],
config['cookie']['expiry_days'],
config['preauthorized']
)

if st.session_state["authentication_status"]:
    st.sidebar.write(f'UserID : **{str.upper(st.session_state["username"])}**') #name
    authenticator.logout('Logout', 'sidebar')
    # st.title('Some content')
    # elif st.session_state["authentication_status"] == False:
    #     st.error('Username/password is incorrect')
    # elif st.session_state["authentication_status"] == None:
    #     st.warning('Please enter your username and password')
 
  
  
  

  
    # -----------------------------------------
    ## Judgement after setting new criteria
    #------------------------------------------

    # demo_image = st.checkbox('Judgement after setting new criteria')
    # if demo_image: 
    #   st.text("waiting add image")

    # -----------------------------------------
    ## Select image Path in server
    #------------------------------------------
        
    ## file_image by filter 
    cols1, cols2 = st.columns([1 ,2])
    with cols1: 
        st.subheader("**:green[Update image path :]**" )
        current_today = datetime.now()
        current_folder = str(current_today.year) + '-' + str(current_today).split("-")[1]
        current_subfolder =  str(str(current_today).split(" ")[0]).split("-")[2] + '-' + str(current_folder).split("-")[1] + '-'+ str(current_folder).split("-")[0][2:4]

        # st.write(str(current_today).split("-")[1])
        # st.write(current_folder)
        # Path: \\192.168.3.247\DataFolder\980Dual\2023\2023-09\04-09-23

        # name_path = str("//192.168.3.247/DataFolder/980Dual/")
        name_path = str("image_LD/")
        if name_path:
            name_folder = st.text_input("Folder:  ", value="Path_Folder")
            if ((name_folder != "Path_Folder") & (name_folder == "")):
                st.write("Fill the word **:red[Path_Folder]** after that press Enter")
                name_folder = str(current_today.year) + '/' + current_folder + '/' + current_subfolder
                st.write("Example of Path for use :", name_folder)
        else:
            name_folder = str(current_today.year) + '/' + current_folder + '/' + current_subfolder
            
        name_file = st.text_input("Serial No. : (Fill number or scan Barcode)" , value= "image")
        
        if ((name_file != "image") & (name_file != "")):
            st.write("Fill the word **:red[image]** after that press Enter")
            image = st.camera_input("Show QR code")
            bytes_data = image.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            detector = cv2.QRCodeDetector()
            name_file, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)
            # name_file = st.text_input("Serial No. : " , value= "DU233BCY03")  
        else:
            name_file = "image"
        
        
        name_image_1 = str(name_path) + str(name_folder) + "/" + str(name_file) +'-1.jpg'
        name_image_2 = str(name_path) + str(name_folder) + "/" + str(name_file) +'-2.jpg'
        
    with cols2: 
        cols3, cols4 = st.columns([1 ,1])
        with cols3:
            st.text("")
            uploaded_file_1 = st.file_uploader("Image of LD 1", accept_multiple_files=False)
            if uploaded_file_1:
                # name_image_1 = uploaded_file_1
                name_image_1 = str(name_path) + str(name_folder) + "/" + str(uploaded_file_1.name)  
            else:
                st.info('This is a waiting for image', icon="ℹ️")
        with cols4:
            st.text("")
            uploaded_file_2 = st.file_uploader("Image of LD 2", accept_multiple_files=False)
            if uploaded_file_2:
                # name_image_2 = uploaded_file_2
                name_image_2 = str(name_path) + str(name_folder) + "/" + str(uploaded_file_2.name)   
            else:
                st.info('This is a waiting for image', icon="ℹ️")

    ## Prepare Image list for ...
    list_img = [name_image_1, name_image_2]
    # st.markdown(list_img)

    st.markdown("_________________________________________________")





    # Criteria value
    criteria = st.toggle('Setting criteria value', value=False)
    if criteria: 
        with st.expander(':red[Must be setting New criteria ...]',expanded=False):
        #   with st.expander('',expanded=False):
    
            cols1, cols2 = st.columns([1 ,1])
            with cols1:
            # st.text("New criteria")
        
                def UpdateSliderToNumeric_contrast():
                    st.session_state.num_contrast = st.session_state.sli_contrast
                def UpdateNumericToSlider_contrast():
                    st.session_state.sli_contrast = st.session_state.num_contrast
                
                def UpdateSliderToNumeric_surface():
                    st.session_state.num_surface = st.session_state.sli_surface
                def UpdateNumericToSlider_surface():
                    st.session_state.sli_surface = st.session_state.num_surface 
                    
                def UpdateSliderToNumeric_conner():
                    st.session_state.num_conner = st.session_state.sli_conner
                def UpdateNumericToSlider_conner():
                    st.session_state.sli_conner = st.session_state.num_conner 
                    
                def UpdateSliderToNumeric_area():
                    st.session_state.num_area = st.session_state.sli_area
                def UpdateNumericToSlider_area():
                    st.session_state.sli_area = st.session_state.num_area 
          
        
                cols3, cols4, cols5, cols6 = st.columns([0.5 ,12, 3, 1])
                with cols4:
                    # st.text("#1: contrast value is ")
                    contrast_images = st.slider("#1: contrast value is ", min_value=0, max_value=255, step=1, on_change=UpdateSliderToNumeric_contrast, key="sli_contrast")
                    # st.text("#2: surface LD chip value is ")
                    surface_LDchip = st.slider("#2: surface LD chip value is ", step=1, on_change=UpdateSliderToNumeric_surface, key="sli_surface")
                    # st.text("#3: area LD chip value is ")
                    area_LDchip = st.slider("#3: area LD chip value is ", step=1, on_change=UpdateSliderToNumeric_area, key="sli_area")
                    # st.text("#4: conner LD chip value is ")
                    conner_LDchip = st.slider("#4: conner LD chip value is ", step=1, on_change=UpdateSliderToNumeric_conner, key="sli_conner")


                    st.text("""
                            ----------- Criteria in paper -----------
                                #1: contrast value is 210
                                #2: surface LD chip value is 50
                                #3: area LD chip value is 50
                                #4: conner LD chip value is 87.50
                                #5: AR area value is 0
                            """)
        
                with cols5:
                    # st.markdown(contrast_images)
                    contrast_images = st.number_input(" ", min_value=0, max_value=255, step=1, on_change=UpdateNumericToSlider_contrast, key="num_contrast")
                    st.text("")
                    # st.markdown(surface_LDchip)
                    surface_LDchip = st.number_input(" ", step=0.5, on_change=UpdateNumericToSlider_surface, key="num_surface")
                    st.text("")
                    # st.markdown(area_LDchip)
                    area_LDchip = st.number_input(" ", step=0.5, on_change=UpdateNumericToSlider_area, key="num_area") 
                    st.text("")
                    # st.markdown(conner_LDchip)
                    conner_LDchip = st.number_input(" ", step=0.5, on_change=UpdateNumericToSlider_conner, key="num_conner")
        
          
            with cols2:
                
                # -----------------------------------------
                ## Show image for setting new criteria
                #------------------------------------------
                
                uploaded_files = st.file_uploader("Choose an other image from upload file", accept_multiple_files=False)
                st.markdown("**:green[Example adjust contrast result :]**" )
        
                if uploaded_files:
                    image = Image.open(uploaded_files)
                
                    cols11, cols12 = st.columns([1 ,1])
                    with cols11:
                        
                        # -----------------------------------------
                        ## Image Procesing pre-process1
                        #------------------------------------------
                        
                        st.text('IMG_1: Pre-process')
                        st.image(image, caption= uploaded_files.name)
                        
                        # Saves
                        img = image.save("img.jpg")

                        # OpenCv Read
                        img_arr = cv2.imread("img.jpg")
                        img_arr = cv2.resize(img_arr, (1600, 1200))
                        resized_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                        _,img_thresh = cv2.threshold(resized_arr , contrast_images, 255, cv2.THRESH_BINARY)
                        
                        
                        new_image = img_thresh

                        # -----------------------------------------
                            ## Show post-process image1
                        #------------------------------------------
                
                    with cols12:
                        st.text('IMG_2: Post-process')
                        st.image(new_image, caption= uploaded_files.name)
        
                elif uploaded_files == None:
                    
                    switch_img = st.toggle('switch LD image from 1 to 2')
                    if switch_img:
                        name_imgLD = name_image_1
                        name_pic = str(name_file) +'-2.jpg'
                    else:
                        name_imgLD = name_image_2
                        name_pic = str(name_file) +'-1.jpg'
                        
                    image = Image.open(name_imgLD)
                    
                    # -----------------------------------------
                        ## Image Procesing pre-process2
                    #------------------------------------------
                    
                    cols11, cols12 = st.columns([1 ,1])
                    with cols11:
                        st.text('IMG_1: Pre-process')
                        st.image(image, caption= name_pic)   
                
                    # Saves
                    img = image.save("img.jpg")

                    # OpenCv Read
                    img_arr = cv2.imread("img.jpg")
                    img_arr = cv2.resize(img_arr, (1600, 1200))
                    resized_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                    _,img_thresh = cv2.threshold(resized_arr , contrast_images, 255, cv2.THRESH_BINARY)
                    
                    
                    new_image = img_thresh

                    # -----------------------------------------
                        ## Show post-process image2
                    #------------------------------------------
                    
                    with cols12:
                        st.text('IMG_2: Post-process')
                        st.image(new_image, caption= name_pic)
            
    else:
        st.info('Setting criteria after add image both 2 LD', icon="ℹ️") #⚠️













    # if list_img[0] == "//192.168.3.247/DataFolder/980Dual/Path_Folder/image-1.jpg":
    # if list_img[0] == (str(name_path) + str(name_folder)) + "/image-1.jpg":
    if criteria == False:
        
        Surface_limit = 0
        LDarea_limit = 0
        ConnerArea_limit = 0
        ARArea_limit = 0
    else:

        # # -----------------------------------------
        #   ## Start Judgement
        # #------------------------------------------

        ## new criterail value for judge
        Contrast_images = contrast_images
        Surface_limit = surface_LDchip
        LDarea_limit = area_LDchip
        ConnerArea_limit = conner_LDchip
        ARArea_limit = 0

        #--------------------------------------

        file_image = algorithm.load_data(list_img)

        rows = int(len(file_image))
        LDBox_img = []
        OUTLDBox_img = []
        post_img = []
        case1_ans = [] ;case2_ans = []
        case3_ans = [] ;case4_ans = []

        # st.text(file_image)
        # st.markdown(len(file_image))
        # st.image(file_image[0][1], caption= file_image[0][2])
        # st.image(file_image[1][1], caption= file_image[0][2])
        # st.text(rows)

        ## level of threshold for seperate limit bightness value : (Hight = 200 , noemal =240) 
        for j in range (int(rows)):
            
            print( " name_image:", j, " :",file_image[j][2])
            image = file_image[j][1].copy()
            
            # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _,img_thresh = cv2.threshold(image , Contrast_images, 255, cv2.THRESH_BINARY)

            kernel = np.ones((50,50),np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
            contours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            if len(contours[0]) != 0:
                largest_contour = max(contours[0], key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)

                # Draw the rectangle on the template image
                box_img = cv2.boxPoints(rect)
                box_img = np.int0(box_img)

                if box_img[1][0] < 800:
                    ##PNG
                    box_LT = box_img[1]
                    box_LB = box_img[0]
                    box_RT = box_img[2]
                    box_RB = box_img[3]
                    
                else:
                    ##JPG
                    box_LT = box_img[0]
                    box_LB = box_img[3]
                    box_RT = box_img[1]
                    box_RB = box_img[2]

                ## 
                theta = math.atan(np.abs(box_RT[1]-box_LT[1])/np.abs(box_RT[0]-box_LT[0]))

                if box_img[1][0] < 1000:
                    mat =  cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2),math.degrees(theta),1)
                else:
                    mat =  cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2),-math.degrees(theta),1)
                rotate_img =cv2.warpAffine(image.copy(),mat,(image.shape[1], image.shape[0]))

                # print(theta, math.degrees(theta))

                ## seperate between solder and LD
                _,img_rotate_gray = cv2.threshold(rotate_img , Contrast_images, 1, cv2.THRESH_BINARY)
                imgy = img_rotate_gray.copy()[0:int(img_rotate_gray.shape[0]) , int(img_rotate_gray.shape[1]/3):int(img_rotate_gray.shape[1]*2/3)]
                sum_y = np.sum(imgy, axis=1)
                posiy = np.where(sum_y>300)
                arrayminy = np.array(posiy[0])
                miny = int(arrayminy[0])
                # plt.plot(sum_y)

                # #img_crop = rotate_img.copy()[startY:endY, startX:endX]
                img_crop = rotate_img.copy()[0:int(miny)+400, 0:rotate_img.shape[1]]

                #--------------------------------------

                # img_gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                ret,thresh_img_crop = cv2.threshold(img_crop,Contrast_images,255,cv2.THRESH_BINARY)
                contours_crop = cv2.findContours(thresh_img_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour_crop = max(contours_crop[0], key=cv2.contourArea)
                rect_crop = cv2.minAreaRect(largest_contour_crop)

                # Draw the rectangle on the template image
                box_img_crop = cv2.boxPoints(rect_crop)
                box_img_crop = np.int0(box_img_crop)

                if box_img_crop[1][0] < 800:
                    ##PNG
                    box_LT_crop = box_img_crop[1]
                    box_LB_crop = box_img_crop[0]
                    box_RT_crop = box_img_crop[2]
                    box_RB_crop = box_img_crop[3]
                    
                else:
                    ##JPG
                    box_LT_crop = box_img_crop[0]
                    box_LB_crop = box_img_crop[3]
                    box_RT_crop = box_img_crop[1]
                    box_RB_crop = box_img_crop[2]

                # print(box_img)

                line_VL = box_LB_crop[1]-box_LT_crop[1]
                line_VR = box_RB_crop[1]-box_RT_crop[1]
                line_HT = box_RT_crop[0]-box_LT_crop[0]
                line_HB = box_RB_crop[0]-box_LB_crop[0]
                

                input_pts = np.float32([box_LT_crop, box_LB_crop, box_RB_crop, box_RT_crop])
                output_pts = np.float32([[0,0], [0, line_VL],[line_HB, line_VR],[line_HT, 0]])
                    
                # Compute the perspective transform M
                M = cv2.getPerspectiveTransform(input_pts,output_pts)

                # Apply the perspective transformation to the image
                output_img = cv2.warpPerspective(thresh_img_crop.copy(),M,(np.min([line_HT,line_HB]), np.min([line_VL,line_VR])),flags=cv2.INTER_LINEAR)
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGRA2BGR)
                # print(output_img.shape[0])
                
                # write box region of interest
                # st.image(cv2.drawContours(cv2.cvtColor(rotate_img, cv2.COLOR_BGR2RGB), [box_img_crop], 0, (0, 0, 255), 10))
                post = cv2.drawContours(cv2.cvtColor(rotate_img, cv2.COLOR_BGR2RGB), [box_img_crop], 0, (127, 0, 127), 5)
                post_img.append([j, post])

                #--------------------------------------
                
                img_check1 = output_img.copy()[0:int(output_img.shape[0]) , 0:int(output_img.shape[1])]
                sum_x1 = np.sum(img_check1, axis=1)/255.
                
                if (output_img.shape[0] > 250) & (int(sum_x1[int(output_img.shape[0]/2)][0]) > 1000): 
                    
                    # x axis
                    imgxl = output_img.copy()[0:int(output_img.shape[0]) , 0:int(output_img.shape[1]/2)]
                    sum_xl = np.sum(imgxl, axis=0)/255.
                    posixl = np.where(sum_xl>200)
                    arraymaxxl = np.array(posixl[0])
                    maxxl = int(arraymaxxl[0])

                    imgxr = output_img.copy()[0:int(output_img.shape[0]) , int(output_img.shape[1]/2):output_img.shape[1]]
                    sum_xr = np.sum(imgxr, axis=0)/255.
                    posixr = np.where(sum_xr>200)
                    arraymaxxr = np.array(posixr[0])
                    maxxr = int(arraymaxxr[-1])

                    # y axis
                    imgyl = output_img.copy()[0:int(output_img.shape[0]/2) , 0:int(output_img.shape[1])]
                    sum_yl = np.sum(imgyl, axis=1)/255.
                    posiyl = np.where(sum_yl>800)
                    arrayminyl = np.array(posiyl[0])
                    minyl = int(arrayminyl[1])

                    imgyr = output_img.copy()[int(minyl):int(minyl)+350 , 0:int(output_img.shape[1])]
                    sum_yr = np.sum(imgyr, axis=1)/255.
                    posiyr = np.where(sum_yr>800)
                    arraymaxyr = np.array(posiyr[0])
                    maxyr = int(arraymaxyr[-1])

                    # shown image
                    new_crop = output_img.copy()[int(minyl):int(minyl)+int(maxyr) , int(maxxl):int(output_img.shape[1]/2)+int(maxxr)]
                    new_img = cv2.resize(new_crop, (350, 100))

                    img_check2 = new_crop.copy()[0:int(new_crop.shape[0]) , int(new_crop.shape[1]/8):int(new_crop.shape[1]/4)]
                    sum_x2 = np.sum(img_check2, axis=0)/255.
                    
                    
                    if (int(np.max(sum_x2)) > 250):
                        LDBox_img.append([file_image[j][0], new_img, "normal"])
                        
                        print( " name_image:", j, " :",file_image[j][2], "check case1")        
                        case1_result = algorithm.CheckSurfaceLDArea(new_img, Surface_limit, Contrast_images)
                        case1_ans.append([j, case1_result])
                        

                        print( " name_image:", j, " :",file_image[j][2], "check case2")
                        case2_result = algorithm.CheckLDArea(new_img, LDarea_limit, Contrast_images) 
                        case2_ans.append([j, case2_result])              

                    
                        print( " name_image:", j, " :",file_image[j][2], "check case3")
                        case3_result = algorithm.CheckConnerArea(new_img, ConnerArea_limit, Contrast_images) 
                        case3_ans.append([j, case3_result])       

            
                        print( " name_image:", j, " :",file_image[j][2], "check case4")
                        case4_result = algorithm.CheckARArea(new_img, ARArea_limit, Contrast_images)
                        case4_ans.append([j, case4_result]) 
                        
                    else:
                        LDBox_img.append([file_image[j][0], new_img, "abnormal"])
                        OUTLDBox_img.append([file_image[j][0], new_img, j])
                        
                        print( " name_image:", j, " :",file_image[j][2], "check case4/3 - abmormal")
                        case4_2abnormal_result = algorithm.CheckARArea(new_img, ARArea_limit, Contrast_images)
                        case4_ans.append([j, case4_2abnormal_result])
                        case1_ans.append([j, "-"])
                        case2_ans.append([j, "-"])
                        case3_ans.append([j, "-"])
                
                else:
                    LDBox_img.append([file_image[j][0], cv2.resize(output_img, (350, 100)), "abnormal"])
                    OUTLDBox_img.append([file_image[j][0], cv2.resize(output_img, (350, 100)), j])
                    
                    print( " name_image:", j, " :",file_image[j][2], "check case4/2 - abmormal")  
                    case4_1abnormal_result = algorithm.CheckARArea(cv2.resize(output_img, (350, 100)), ARArea_limit, Contrast_images)
                    case4_ans.append([j, case4_1abnormal_result]) 
                    case1_ans.append([j, "-"])
                    case2_ans.append([j, "-"])
                    case3_ans.append([j, "-"]) 
            
            else:
                LDBox_img.append([file_image[j][0], cv2.resize(image, (350, 100)), "abnormal"])
                OUTLDBox_img.append([file_image[j][0], cv2.resize(image.copy(), (350, 100)), j])
                print( " name_image:", j, " :",file_image[j][0], "check case4/1 - abmormal") 
                case1_ans.append([j, "-"])
                case2_ans.append([j, "-"])
                case3_ans.append([j, "-"])
                case4_ans.append([j, "-"])
        
        ## ------------------------------------------------------------------

        labels  = ['Name_image', 'Case AR area', 'Case surface area', 'Case LD area', 'Case conner area','Result judgement', 'Remark'] 
        result_LD = [] 
        result_AR = []
        result_judge = []

        for ij in range (len(file_image)):
            # st.write(ij)
            
            if ((case1_ans[ij][1] == 'OK') & (case2_ans[ij][1] == 'OK') & (case3_ans[ij][1] == 'OK')):
                result_LD.append([ij ,'OK'])
            else:
                result_LD.append([ij, 'NG'])
                
            if (case4_ans[ij][1] == 'NG'):  
                result_judge.append([ij, 'NG'])
                result_AR.append([ij, 'NG'])
            else:
                result_judge.append([ij, result_LD[ij][1]])
                result_AR.append([ij, 'OK'])
            
            ## Remark confrim LD
            if (LDBox_img[ij][2] == "normal"):
                remark = '-'
            else:    
                remark = 'Wait operator judge'
            
    
    
    
    
    
    
    
    
    
            
        # -----------------------------------------
        ## Display result Judgement
        #------------------------------------------

        if(Surface_limit == 0) | (LDarea_limit == 0) | (ConnerArea_limit == 0):
            st.info('Result will display after setting criteria ', icon="ℹ️")
            
        elif (Surface_limit != 0) & (LDarea_limit != 0) & (ConnerArea_limit != 0):
            st.subheader("**:violet[Display images :]**")
            st.text("")

            cols003, cols5, cols6, cols7, cols8 = st.columns([0.1, 10 ,10, 0.5, 10])
            with cols5:    
                if (str(name_file) == ''):
                    st.info('This is a waiting for image', icon="ℹ️")
                else:
                    IMG_1 = Image.open(name_image_1)
                    st.image(IMG_1, caption= name_file +'-1.jpg')

            with cols6:
                if (str(name_file) == ''):
                    st.info('This is a waiting for image', icon="ℹ️")
                else:
                    IMG_2 = Image.open(name_image_2)
                    st.image(IMG_2, caption= name_file +'-2.jpg')

            with cols8:
                st.text("---- Judgement Result ----")
                cols9, cols10, cols11, cols002 = st.columns([0.5, 5, 1, 1])
                with cols10:
                    st.text("""
                            Result of LD chip 1
                            Result of LD chip 2
                            """)
                
                with cols11:
                    st.markdown(result_judge[0][1] + "        " + "\n" +result_judge[1][1])
            
            
                detail_LD1 = st.checkbox('more detail of LD chip')
                if detail_LD1: 
                    with st.expander(':gray[------------ Result judgment ------------]',expanded=True):
                        cols9, cols10 = st.columns([3, 2])
                        with cols9:
                            st.text("")
                            st.text("""
                                Topic
                                #1: surface LD chip 
                                #2: area LD chip   
                                #3: conner LD chip 
                                #4: AR area        
                            """)
                    
                        with cols10:
                            st.text("LD-1     LD-2")
                            st.text(case1_ans[0][1]+ "        " +case1_ans[1][1]+"\n"+ case2_ans[0][1]+ "        " +case2_ans[1][1]+"\n"+ case3_ans[0][1]+ "        " +case3_ans[1][1]+"\n"+ case4_ans[0][1]+ "        " +case4_ans[1][1])
            
                st.markdown("_______________________________________")
            
                cols12, cols13, cols14 = st.columns([0.1, 8, 10])
                if (result_judge[0][1] == 'OK') & (result_judge[1][1] == 'OK'):
                    result_LD = "OK"
                else:
                    result_LD = "NG"
                
                with cols13:
                    st.markdown("**Total judgement**")
                    if st.button("SAVE"):
                        df1 = pd.read_csv("./result_judgement.csv")
                        current_today = datetime.now()
                        current_file = str(current_today).split(" ")[0]
    
                        dict = {'ID': [str.upper(st.session_state["username"])],
                                'username':[name_file],
                                'contrast': [Contrast_images],
                                'surface': [Surface_limit],
                                'LDarea': [LDarea_limit],
                                'connerArea': [ConnerArea_limit],
                                'ARArea': [ARArea_limit],
                                'LD chip 1     ': [case1_ans[0][1]+"/"+ case2_ans[0][1]+"/"+ case3_ans[0][1]+"/"+ case4_ans[0][1]],
                                'LD chip 2     ': [case1_ans[1][1]+"/"+ case2_ans[1][1]+"/"+ case3_ans[1][1]+"/"+ case4_ans[1][1]],
                                'result': [result_LD],
                                'remark': [remark],
                                'updatedate': [current_file]
                            }
                        df2 = pd.DataFrame(dict)
                        df3 = pd.concat([df1, df2], ignore_index = True)
                        st.write(df3) 
                        df3.to_csv('result_judgement.csv', index = False)
                
                with cols14:
                    if (result_LD == "OK"):
                        result = '<p style="font-family:sans-serif; color:blue; font-size: 90px;"> OK </p>'
                        st.markdown(result , unsafe_allow_html=True)
                    else:
                        result = '<p style="font-family:sans-serif; color:red; font-size: 90px;"> NG </p>'
                        st.markdown(result , unsafe_allow_html=True)
                
        
    
        
            # st.markdown("**:black[LD Area check:]**")      
            if detail_LD1: 
        
                with cols5:
                    if (str(name_file) == ''):
                        st.info('This is a waiting for image', icon="ℹ️")
                    else:
                        IMG_3 = post_img[0][1]
                        st.image(IMG_3, caption= name_file +'-1.jpg')

                with cols6:
                    if (str(name_file) == ''):
                        st.info('This is a waiting for image', icon="ℹ️")
                    else:
                        IMG_4 = post_img[1][1]
                        st.image(IMG_4, caption= name_file +'-2.jpg')
            
    
        
        st.markdown("_________________________________________________")

 

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')
    

# -----------------------------------------
  ## Remark select page
#------------------------------------------

# Rest of the page
# st.markdown("# Operator Judgement")
# st.sidebar.subheader(" ")


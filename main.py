
def measure_props(summary_dir, dir_list):
 all_files = []
 #dir_list= dir_list[12:14] # for debugging
 #Loops through public directories and scans for folders with roi file type
 for dir_ in dir_list:
 #dir_ = dir_list[7] # for debugging
 # load all tiff movie files that end in _N_roi.tif (N == a number)
 files = glob(dir_[5] + '/*.roi')
 """iterating through public directories (what does dir_i? mean) and stores in some_files"""
 for dir_i in dir_list:
 some_files = glob(dir_i[5] + '/*.roi') # change to _roi.tif later
 """adds roi string, and if there are any files stored in some_files, add stored element to the list in all_files"""
 if (len(some_files) > 0):
 all_files.append(some_files)
 some_files = []
 else:
 continue

 # skip if there are no rois
 # counting # of directories stored in all_files and if there is less than one, meaning none, then skip
 if (len(all_files) < 1):
 continue
 all_spools = pd.DataFrame(all_files)
 spools = all_spools.apply(pd.Series).stack().unique()
 lspools = spools.tolist()

 # obtain number of samples
 sep_files = []
 # looping through unique files (removing duplicate) and generate range returning list of #'s corresponding to file
 # for each loop num assumes a new value
# then compile to regular expression and seperate from # and  the add sep_file's 3rd element to target file sep_files
 for num in range(0, len(lspools)):
 sep_file = re.compile(r"[\\]").split(lspools[num])
 sep_files.append(sep_file[2])
 sep_file = []

 counts, values = pd.Series(sep_files).value_counts().values, pd.Series(sep_files).value_counts().index
 df_results = pd.DataFrame(list(zip(values, counts)), columns=["value", "count"])
 sample_counts_dir = df_results.loc[df_results["value"] == dir_[3]]

 # Find ROI file
 # single ROI
 # count # of roi files in roi_list and if length is 1, then there isn't a zipped file
 # if the length is less than 1, then there are no files in this list, so it scans for a zipped file
 # for lengths beyond these conditions, it will print message and it will continue to the next iteration
 roi_list = glob(dir_[5] + '/*.roi')
 if (len(roi_list) == 1):
 roi_is_zip = False
 elif (len(roi_list) < 1):
 # more than one ROI
 roi_list = glob(dir_[5] + '/*.zip')
 roi_is_zip = True
 else:
 print("Missing ROI file for movie file")
 continue

# scanning for roi_file in roi_list
 for roi_file in roi_list:
 # load the ROI and extract the coords
 # make a mask from the coords
 # create folder for this cropped out cell
 # use mask to save the cropped movie file
 if (roi_is_zip):
 roi_points = read_roi.read_roi_zip(roi_file)
 """if the zipped file DNE, then it will store the opened roi file that is read in binary mode as 
 fobj and then read fobj and store in roi_points, and then matches file formats as if it is a zip file"""
 else:
 fobj = open(roi_file, 'r+b')
 roi_points = read_roi.read_roi(fobj)
 roi_points = [roi_points, ] # make format same as if its a zip

 # ROI is bounding box, make mask
 # go through each frame and apply masks, save cropped movie file
 #roi_points_test = roi_points[1] # for debugging
 """based on file coords, roi_points = roi_count, bbox_points = second-coord, add # values to refer to file types"""
 for roi_count, bbox_points in enumerate(roi_points):#enumerate(roi_points_test):
 # roi_count = 1 # for debugging
 file = []
 # for loop is scanning for unique files with desired extensions for storing them in roifile
 for ext in ('/*.roi', '/*.zip'):
 roifile = glob(dir_[5] + ext)
 """if there are any files in roifile, joins the files as one iterable item and adds it to roifile, then prints message
 for each file"""
 if (len(roifile) > 0):
 file.append(''.join(roifile))
 file = ''.join(file)
 print('Processing ', file[:-4] + '_' + str(roi_count + 1))
 file_root = os.path.split(file)[1][:-4]
 sep_file_ = re.compile(r"[\\]").split(file)
 roi_folder = file_root + '_' + str(roi_count + 1)
 date_file = sep_file_[1][:-7]

# if the file path exists then save read/retrieved data from .csv file to df_foci
 test = dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status.csv'
 test_ = os.path.isfile(dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status.csv') == True
 if (os.path.isfile(dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status.csv') == True):
 df_foci = pd.read_csv(dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status.csv')
 else:
 continue

 properties_full = []
 #df_foci_test = df_foci.iloc[8:10]
 #iterates through each row of the dataframe and prints message for each; stores value from row dictionary under
 # column "focus_label" to local focus_label variable
 for index, row in df_foci.iterrows():
 #print(index), 28
 #index = 70
 #row = df_foci.iloc[26]
 print('Focus label: ' + str(row['focus_label']))
 print('Index: ' + str(index))
 properties = []
 test = roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_proc_10_maxproj_roi' + str(roi_count + 1) + '.tif'

 # import images
 # max proj
 # no outlines
 # each chunk reads the file image and stores it in new folder
 img = io.imread(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_proc_10_maxproj_roi' + str(roi_count + 1) + '.tif') # import max projection image
 # outlines
 img_outlines = io.imread(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_procfocusDetectionCheck_Full.tif')
 # stores max proj images of raw data to img and img_raw and raw images to img_outlines and img_raw_outlines
 # no outlines
 img_raw = io.imread(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_proc_10_rawmaxproj_roi' + str(roi_count + 1) + '.tif') # import max projection image
 # outlines
 img_raw_outlines = io.imread(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_rawfocusDetectionCheck_Full.tif')

 # extract focus bounding box
 bbox = row['bbox'].split('(', 2)
 test1 = bbox[1].split(')', 2)
 test2 = test1[0].split(',', 4)
 results = list(map(int, test2))

 pixel_border = 3
 r0 = results[0] - pixel_border
 r1 = results[2] + pixel_border
 c0 = results[1] - pixel_border
 c1 = results[3] + pixel_border
 img_test = img[results[0]:results[2], results[1]:results[3]]
 img_test = (img_test - np.min(img_test)) / (np.max(img_test) - np.min(img_test)) # scale from 0 to 1
 img_test = np.multiply(img_test, 255) # scale from 0 to 255
 img_test = img_test.astype(np.float32) # 'uint32'
 # pixel no
 len_img = img_test.size

 # A) BLUR DETECTION
 # A.0) blur detection with variance of laplacian
 # 0 mean proj single foci
 # stores laplacian transformed image with calculated variance to var_laplacian and saves img in roi file
 var_laplacian = cv2.Laplacian(img_test, cv2.CV_32F).var() # variance of laplacian
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_0vl_test_.tif', img_test) # save img
 laplacian = cv2.Laplacian(img_test, cv2.CV_32F) # laplacian
 # apply laplacian operator to detect edges within img and save locally without variance
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_0l_test_.tif', laplacian) # save img

 # 1 gaussian blur 0 of 0
 img_test_gb0 = cv2.GaussianBlur(img_test, (3, 3), 0)
 var_laplacian_gb0 = cv2.Laplacian(img_test_gb0, cv2.CV_32F).var() # variance of laplacian
 #save img from roifile with computer vision 2 gaussian blur + laplacian applied to region (3,3) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_1vlgb0_test.tif', img_test_gb0) # save img
 laplacian_gb0 = cv2.Laplacian(img_test_gb0, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb0 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_1l_gb0_test.tif', laplacian_gb0) # save img

 # 2 gaussian blur 1 of 0
 img_test_gb1 = cv2.GaussianBlur(img_test, (5, 5), 0)
 var_laplacian_gb1 = cv2.Laplacian(img_test_gb1, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (5,5) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_2vlgb1_test.tif', img_test_gb1) # save img
 laplacian_gb1 = cv2.Laplacian(img_test_gb1, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb1 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_2l_gb1_test.tif', laplacian_gb1) # save img


 # 3 gaussian blur 2 of 0
 img_test_gb2 = cv2.GaussianBlur(img_test, (7, 7), 0)
 var_laplacian_gb2 = cv2.Laplacian(img_test_gb2, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (7,7) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_3vlgb2_test.tif', img_test_gb2) # save img
 laplacian_gb2 = cv2.Laplacian(img_test_gb2, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb2 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_3l_gb2_test.tif', laplacian_gb2) # save img


 # 4 gaussian blur 3 of 0
 img_test_gb3 = cv2.GaussianBlur(img_test, (9, 9), 0)
 var_laplacian_gb3 = cv2.Laplacian(img_test_gb3, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (9,9) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_4vlgb3_test.tif', img_test_gb3) # save img
 laplacian_gb3 = cv2.Laplacian(img_test_gb3, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb3 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_4l_gb3_test.tif', laplacian_gb3) # save img

 # A.1) blur detection using measure function
 blur_2 = measure.blur_effect(img_test)

 # B) shannon entropy
 shannon_entropy = measure.shannon_entropy(img_test)

 # C) stats
 focus_mean = np.mean(img_test)
 focus_med = np.median(img_test)
 focus_sum = np.sum(img_test)
 focus_std = np.std(img_test)
 focus_var = np.var(img_test)
 focus_max = np.max(img_test)
 focus_min = np.min(img_test)

 # D) compactness
 focus_area = row['area']
 focus_perim = row['perimeter']

# checking for focus perimeter (refering to value in "perimeter column" through row - dictionary)
 # if focus perimeter is too small then index/row gets dropped and prints message "focus too small"
 if (focus_perim < 1):
 df_foci.drop(index, inplace=True)
 print('Focus is too small. Focus dropped.')
 continue

 # D.0) Polsby-Popper
 compactness_pp = (4 * pi * focus_area) / (focus_perim * focus_perim)

 # D.1) Schwartzberg
 compactness_s = 1 / (focus_perim / (2 * pi * sqrt(focus_area / pi)))

 # E) other geometric properties
 # 1.2 connected component labeling
 # img_mask.reshape(n_rows, n_cols)
 l_, n_ = mh.label(img_test, np.ones((3, 3), bool)) # binary_closed_hztl_k
 #saves the roi_file with string applied to extract row that contains focus_label???
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_CCL_mag.tif', l_)

 # 1.3 measure region properties
 rs_k = regionprops(l_)
 im_props = regionprops(l_, intensity_image=img_test)
 results = []

 # only append focus properties if single object is detected in magnitude spectrum
 # if a single object is detected in roi, then proceed with nested for loop that applies image properties for blobs
 # if not 1, then proceeds to else
 if (len(im_props) == 1):
 for focus_blob in im_props:
 # if (focus_blob.area > 1000):
 # continue
 # blob = im_props[30]
 properties.append(focus_blob.euler_number)
 properties.append(focus_blob.solidity)
 stop = 1
 #drop removes dataframe index column and prints "focus dropped" message if a single object has not been detected
 else:
 df_foci.drop(index, inplace=True)
 print('Focus bbox has more or less than 1 object. Focus dropped.')
 continue

 # F) FFT Magnitude properties
 # FFT = fast fourier transform - tool for analyzing and measuring signals from plug-in data acquisition devices
 # ft
 f = np.fft.fft2(img_test)
 # magnitude: ft
 magnitude_spectrum_ft = 20 * np.log(np.abs(f))
 # shift ft
 fshift = np.fft.fftshift(f)
 # magnitude: shifted ft
 magnitude_spectrum_fshift = 20 * np.log(np.abs(fshift))
 magnitude_spectrum_fshift_32F = magnitude_spectrum_fshift.astype(np.float32) # 'uint16'

 # F.A) Blur detection
 # F.A.0) Blur detection, Variance of Laplacian
 fft_var_laplacian = cv2.Laplacian(magnitude_spectrum_fshift_32F, cv2.CV_32F).var() # variance of laplacian
 # save img as 32-bit with magnitude spectrum plot describing frequency and intensity (fourier transform)
 # different extensions to identify three locations where img w variance is saved, applying mag. shift
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_5mag_.tif', magnitude_spectrum_fshift_32F) # save img
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_5vlm_.tif', magnitude_spectrum_fshift_32F) # save img
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_input5mag_.tif', img_test)

 # F.A.1) blur detection using measure function
 fft_blur_2 = measure.blur_effect(magnitude_spectrum_fshift_32F)

 # F.B) shannon entropy
 fft_shannon_entropy = measure.shannon_entropy(magnitude_spectrum_fshift_32F)

 # F.C) Stats
 fft_mean = np.mean(magnitude_spectrum_fshift_32F)
 fft_med = np.median(magnitude_spectrum_fshift_32F)
 fft_sum = np.sum(magnitude_spectrum_fshift_32F)
 fft_std = np.std(magnitude_spectrum_fshift_32F)
 fft_var = np.var(magnitude_spectrum_fshift_32F)
 fft_max = np.max(magnitude_spectrum_fshift_32F)
 fft_min = np.min(magnitude_spectrum_fshift_32F)

 # F.D) Other Geometric Properties
 # 1.1 OTSU thresholding
 # save mag shift img with computer vision 2 gaussian blur applied to region (3,3) -ksize
 magnitude_spectrum_fshift_32F_gb = cv2.GaussianBlur(magnitude_spectrum_fshift_32F, (3, 3), 0)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_gb3_mag_.tif', magnitude_spectrum_fshift_32F_gb)

 th = filters.threshold_otsu(magnitude_spectrum_fshift_32F_gb)
 # filters.threshold_otsu returns threshold value of img using otsu's method, and stores this value as th
 # greater than comparison returns true if mag shift is greater than th; this sets the threshold + stores it in img_mask
 img_mask = magnitude_spectrum_fshift_32F_gb > th
 # save img_masked image with mag shift/otsu filter applied to region (3,3) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_OTSUthresh_mag.tif', img_mask)

 # 1.2 connected component labeling
 # img_mask.reshape(n_rows, n_cols)
 #mh.label - labels contiguous regions of the image
 #saves img with labels; labelled images are integer images where values correspond to different regions
 l_, n_ = mh.label(img_mask, np.ones((3, 3), bool)) # binary_closed_hztl_k
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_CCL_mag_.tif', l_)

 # 1.3 measure region properties
 rs_k = regionprops(l_)
 im_props = regionprops(l_, intensity_image=magnitude_spectrum_fshift_32F_gb)
 results = []

 # only append focus properties if single object is detected in magnitude spectrum
 # for loop applies properties for items in im_props
 # if there is one item in im_props, then continue with for loop and for each blob apply properties
 # if not 1, then proceeds to else
 if (len(im_props) == 1):
 for blob in im_props:
 # if (blob.area > 1000):
 # continue
 # blob = im_props[30]
 properties.append(blob.orientation)
 properties.append(blob.area)
 properties.append(blob.perimeter)
 properties.append(blob.major_axis_length)
 properties.append(blob.minor_axis_length)
 properties.append(blob.eccentricity)
 properties.append(blob.euler_number)
 properties.append(blob.solidity)

 fft_area = blob.area
 fft_perimeter = blob.perimeter
 # F.E) compactness
 # F.E.0) Polsby-Popper
 fft_compactness_pp = (4 * pi * fft_area) / (focus_perim * fft_perimeter)
 properties.append(fft_compactness_pp)
 # F.E.1) Schwartzberg
 fft_compactness_s = 1 / (fft_perimeter / (2 * pi * sqrt(fft_area / pi)))
 properties.append(fft_compactness_s)
 #drop removes dataframe index column and prints "focus dropped" message if # of items is not 1
 else:
 df_foci.drop(index, inplace=True)
 print('FFT magnitude has more or less than 1 object. Focus dropped.')
 continue

# store first element in properties in area
 # conditional statement that checks if the area of the foci is larger than 1000, if it is, then row is dropped + msg
 area = properties[1]
 if (area > 1000):
 df_foci.drop(index, inplace=True)
 print('FFT magnitude object area is too large. Focus dropped.')
 continue

 # Append rest of fft properties
 properties.append(fft_var_laplacian)
 properties.append(fft_blur_2)
 properties.append(fft_shannon_entropy)
 properties.append(fft_mean)
 properties.append(fft_med)
 properties.append(fft_sum)
 properties.append(fft_std)
 properties.append(fft_var)
 properties.append(fft_max)
 properties.append(fft_min)

 # Append rest of focus properties
 properties.append(var_laplacian)
 properties.append(var_laplacian_gb0)
 properties.append(var_laplacian_gb1)
 properties.append(var_laplacian_gb2)
 properties.append(var_laplacian_gb3)
 properties.append(blur_2)
 properties.append(shannon_entropy)
 properties.append(focus_mean)
 properties.append(focus_med)
 properties.append(focus_sum)
 properties.append(focus_std)
 properties.append(focus_var)
 properties.append(focus_max)
 properties.append(focus_min)
 properties.append(compactness_pp)
 properties.append(compactness_s)
 properties.append(len_img)

# numpy isnan function checks if the input is a NaN (meaning not a number), then returns result as boolean array
 # any tests all array elements along all axis, bc an axis has not been specified in parameters
 # prints message and drops row if NaN is True, then adds properties to properties_full list
 if (np.isnan(properties).any() == True):
 print("Drop focus, contains nan in metrics.")
 df_foci.drop(index, inplace=True)
 continue
 properties_full.append(properties)

#conditional statement: if the # of items in df_foci is less than 1, then prints message and numpy creates array of list
 # saves to new variable properties_full_ar
 if (len(df_foci) < 1):
 print('Focus dataframe empty. Skip this ROI.')
 continue
 properties_full_ar = np.array(properties_full)

 df_foci['euler_number'] = properties_full_ar[:, 0]
 df_foci['solidity'] = properties_full_ar[:, 1]

 df_foci['fft_orientation'] = properties_full_ar[:, 2]
 df_foci['fft_area'] = properties_full_ar[:, 3]
 df_foci['fft_perimeter'] = properties_full_ar[:, 4]
 df_foci['fft_major_axis_length'] = properties_full_ar[:, 5]
 df_foci['fft_minor_axis_length'] = properties_full_ar[:, 6]
 df_foci['fft_eccentricity'] = properties_full_ar[:, 7]
 df_foci['fft_euler_number'] = properties_full_ar[:, 8]
 df_foci['fft_solidity'] = properties_full_ar[:, 9]
 df_foci['fft_compactness_pp'] = properties_full_ar[:, 10]
 df_foci['fft_compactness_s'] = properties_full_ar[:, 11]
 df_foci['fft_var_laplacian'] = properties_full_ar[:, 12]
 df_foci['fft_blur_2'] = properties_full_ar[:, 13]
 df_foci['fft_shannon_entropy'] = properties_full_ar[:, 14]
 df_foci['fft_mean'] = properties_full_ar[:, 15]
 df_foci['fft_med'] = properties_full_ar[:, 16]
 df_foci['fft_sum'] = properties_full_ar[:, 17]
 df_foci['fft_std'] = properties_full_ar[:, 18]
 df_foci['fft_var'] = properties_full_ar[:, 19]
 df_foci['fft_max'] = properties_full_ar[:, 20]
 df_foci['fft_min'] = properties_full_ar[:, 21]

 df_foci['var_lap'] = properties_full_ar[:, 22]
 df_foci['var_lap_gb3'] = properties_full_ar[:, 23]
 df_foci['var_lap_gb5'] = properties_full_ar[:, 24]
 df_foci['var_lap_gb7'] = properties_full_ar[:, 25]
 df_foci['var_lap_gb9'] = properties_full_ar[:, 26]
 df_foci['blur_2'] = properties_full_ar[:, 27]
 df_foci['shannon_entropy'] = properties_full_ar[:, 28]
 df_foci['focus_mean'] = properties_full_ar[:, 29]
 df_foci['focus_med'] = properties_full_ar[:, 30]
 df_foci['focus_sum'] = properties_full_ar[:, 31]
 df_foci['focus_std'] = properties_full_ar[:, 32]
 df_foci['focus_var'] = properties_full_ar[:, 33]
 df_foci['focus_max'] = properties_full_ar[:, 34]
 df_foci['focus_min'] = properties_full_ar[:, 35]
 df_foci['compactness_pp'] = properties_full_ar[:, 36]
 df_foci['compactness_s'] = properties_full_ar[:, 37]
 df_foci['tot_pix_no'] = properties_full_ar[:, 38]

 df_foci.to_csv(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + 'focus_info_full_props.csv')

 stop = 1
 print('Next experiment...')


def measure_props_idr(summary_dir, dir_list):
 all_files = []
 #dir_list = [dir_list[i] for i in [2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23]]
 #dir_list = dir_list[1:3] # for debugging
 #Loops through public directories and scans for folders with roi file type
 for dir_ in dir_list:
 #dir_ = dir_list[6] # for debugging
 # load all tiff movie files that end in _N_roi.tif (N == a number)
 files = glob(dir_[5] + '/*.zip')
 """iterating through public directories (what does dir_i? mean) and stores in some_files"""
 for dir_i in dir_list:
 some_files = glob(dir_i[5] + '/*.zip') # change to _roi.tif later
 """adds zip string, and if there are any files stored in some_files, adds stored element to the list in all_files"""
 if (len(some_files) > 0):
 all_files.append(some_files)
 some_files = []
 else:
 continue

 # skip if there are no rois
 # if there are no files in all_files, then continue with constructing a dataframe and storing it to all_spools
 # then, apply pd.series to create columns in dataframe and stack restructures the multi-level columns to optimize rows
 # then, unique find the unique values in series; series meaning a single row
 # lspools stores series from spools after converting values into a list
 if (len(all_files) < 1):
 continue
 all_spools = pd.DataFrame(all_files)
 spools = all_spools.apply(pd.Series).stack().unique()
 lspools = spools.tolist()

 # obtain number of samples
 # looping through unique files (removing duplicate) and generate range returning list of #'s corresponding to file
 # for each loop num assumes a new value
 # then compile to regular expression and seperate from # and  the add sep_file's 3rd element to target file sep_files
 sep_files = []
 for num in range(0, len(lspools)):
 sep_file = re.compile(r"[\\]").split(lspools[num])
 sep_files.append(sep_file[2])
 sep_file = []

 counts, values = pd.Series(sep_files).value_counts().values, pd.Series(sep_files).value_counts().index
 df_results = pd.DataFrame(list(zip(values, counts)), columns=["value", "count"])
 sample_counts_dir = df_results.loc[df_results["value"] == dir_[3]]

 # Find ROI file
 # single ROI
 dir_files = os.listdir(dir_[5])
 spool_list = sorted(dir_files)
 spool_list_file = spool_list[0]
 roi_list = glob(dir_[5] + '/' + spool_list_file[:-4] + '_roi' + '*.zip') #glob(dir_[5] + '/' + '001-23 53bp10011_roi' + '*.zip')
 roi_is_zip = True

 # scanning for roi_file in roi_list
 for roi_file in roi_list:
 # load the ROI and extract the coords
 # make a mask from the coords
 # create folder for this cropped out cell
 # use mask to save the cropped movie file
 if (roi_is_zip):
 roi_points = read_roi.read_roi_zip(roi_file)
 """if the zipped file DNE, then it will store the opened roi file that is read in binary mode as 
 fobj and then read fobj and store in roi_points, and then matches file formats as if it is a zip file"""
 else:
 fobj = open(roi_file, 'r+b')
 roi_points = read_roi.read_roi(fobj)
 roi_points = [roi_points, ] # make format same as if its a zip

 # ROI is bounding box, make mask
 # go through each frame and apply masks, save cropped movie file
 #roi_points_test = roi_points[1] # for debugging
 """based on file coords, roi_points = roi_count, bbox_points = second-coord, add # values to refer to file types"""
 for roi_count, bbox_points in enumerate(roi_points):#enumerate(roi_points_test):
 # roi_count = 1 # for debugging
 file = []
 # for loop is scanning for unique files with desired extensions for storing them in roifile
 for ext in ('/*.roi', '/*.zip'):
 roifile = glob(dir_[5] + ext)
 """if there are any files in roifile, joins the files as one iterable item and adds it to roifile, then prints message
 for each file"""
 if (len(roifile) > 0):
 file.append(''.join(roifile))
 file = ''.join(file)
 print('Processing ', file[:-4] + '_' + str(roi_count + 1))
 file_root = os.path.split(file)[1][:-4]
 sep_file_ = re.compile(r"[\\]").split(file)
 roi_folder = file_root + '_' + str(roi_count + 1)
 date_file = sep_file_[1][:-7]

# stores to df_foci panda package's read csv
 df_foci = pd.read_csv(dir_[5] + '/' + roi_folder + '/' + 'focus_info_full.csv') #focus_info_full_status.csv
 '''
 # check for csv file with manually classified foci
 if (os.path.isfile(dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status.csv') == True):
 df_foci = pd.read_csv(dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status.csv')
 else:
 continue
 '''

 #df_foci = pd.read_csv(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[4] + 'focus_info_full.csv')
 properties_full = []
 #df_foci_test = df_foci.iloc[26]
 #iterates through each row of the dataframe and prints message for each; stores value from row dictionary under
 # column "focus_label" to local focus_label variable
 for index, row in df_foci.iterrows():
 focus_label = row['focus_label']
 #print(index), 28
 #index = 70
 #row = df_foci.iloc[26]
 print('Focus label: ' + str(row['focus_label']))
 print('Index: ' + str(index))
 properties = []
 # stores in img read roi file as preprocessing + tif/high quality image; importing images, max proj, no outlines
 img = io.imread(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[4] + '_preproc_04_gaussblur_roi'+ str(roi_count + 1) + '.tif') # import max projection image

 # extract focus bounding box
 bbox = row['bbox'].split('(', 2)
 test1 = bbox[1].split(')', 2)
 test2 = test1[0].split(',', 4)
 results = list(map(int, test2))
 img_test = img[results[0]:results[2], results[1]:results[3]]
 img_test = (img_test - np.min(img_test)) / (np.max(img_test) - np.min(img_test)) # scale from 0 to 1
 img_test = np.multiply(img_test, 255) # scale from 0 to 255
 img_test = img_test.astype(np.float32)
 # pixel no
 len_img = img_test.size

 '''
 # save focus in focus folder
 io.imsave(sep_file_[0] + '/' + sep_file_[1] + '/' + 'FocusClasses/' + str(row['status']) + '/' + dir_[0] + '_' + dir_[3] + '_' + dir_[
 4] + '__' + str(sep_file_[4][:-8]) + '_roi' + str(roi_count + 1) +'_focusNo' + str(row['focus_label']) + '.tif', img_test)
 '''

 # A) BLUR DETECTION
 # A.0) blur detection with variance of laplacian
 # 0 mean proj single foci
 # stores laplacian transformed image with calculated variance to var_laplacian and saves img in roi file
 var_laplacian = cv2.Laplacian(img_test, cv2.CV_32F).var() # variance of laplacian
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_0vl_test.tif', img_test) # save img
 laplacian = cv2.Laplacian(img_test, cv2.CV_32F) # laplacian
 # apply laplacian operator to detect edges within img and save locally without variance
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_0l_test.tif', laplacian) # save img

 # 1 gaussian blur 0 of 0
 img_test_gb0 = cv2.GaussianBlur(img_test, (3, 3), 0)
 var_laplacian_gb0 = cv2.Laplacian(img_test_gb0, cv2.CV_32F).var() # variance of laplacian
 #save img from roifile with computer vision 2 gaussian blur + laplacian applied to region (3,3) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_1vlgb0_test.tif', img_test_gb0) # save img
 laplacian_gb0 = cv2.Laplacian(img_test_gb0, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb0 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_1l_gb0_test.tif', laplacian_gb0) # save img

 # 2 gaussian blur 1 of 0
 img_test_gb1 = cv2.GaussianBlur(img_test, (5, 5), 0)
 var_laplacian_gb1 = cv2.Laplacian(img_test_gb1, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (5,5) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_2vlgb1_test.tif', img_test_gb1) # save img
 laplacian_gb1 = cv2.Laplacian(img_test_gb1, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb1 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_2l_gb1_test.tif', laplacian_gb1) # save img


 # 3 gaussian blur 2 of 0
 img_test_gb2 = cv2.GaussianBlur(img_test, (7, 7), 0)
 var_laplacian_gb2 = cv2.Laplacian(img_test_gb2, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (7,7) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_3vlgb2_test.tif', img_test_gb2) # save img
 laplacian_gb2 = cv2.Laplacian(img_test_gb2, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb2 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_3l_gb2_test.tif', laplacian_gb2) # save img

 # 4 gaussian blur 3 of 0
 img_test_gb3 = cv2.GaussianBlur(img_test, (9, 9), 0)
 var_laplacian_gb3 = cv2.Laplacian(img_test_gb3, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (9,9) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_4vlgb3_test.tif', img_test_gb3) # save img
 laplacian_gb3 = cv2.Laplacian(img_test_gb3, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb3 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_4l_gb3_test.tif', laplacian_gb3) # save img

 # A.1) blur detection using measure function
 blur_2 = measure.blur_effect(img_test)

 # B) shannon entropy
 shannon_entropy = measure.shannon_entropy(img_test)

 # C) stats
 focus_mean = np.mean(img_test)
 focus_med = np.median(img_test)
 focus_sum = np.sum(img_test)
 focus_std = np.std(img_test)
 focus_var = np.var(img_test)
 focus_max = np.max(img_test)
 focus_min = np.min(img_test)

 # D) compactness
 focus_area = row['area']
 focus_perim = row['perimeter']

 # checking for focus perimeter (refering to value in "perimeter column" through row - dictionary)
 # if focus perimeter is too small then index/row gets dropped and prints message "focus too small"
 if (focus_perim < 1):
 df_foci.drop(index, inplace=True)
 print('Focus is too small. Focus dropped.')
 continue

 # D.0) Polsby-Popper
 compactness_pp = (4 * pi * focus_area) / (focus_perim * focus_perim)

 # D.1) Schwartzberg
 compactness_s = 1 / (focus_perim / (2 * pi * sqrt(focus_area / pi)))

 # E) other geometric properties
 # 1.2 connected component labeling
 # img_mask.reshape(n_rows, n_cols)
 l_, n_ = mh.label(img_test, np.ones((3, 3), bool)) # binary_closed_hztl_k
 #saves the roi_file with string applied to extract row that contains focus_label???
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_CCL_mag.tif', l_)

 # 1.3 measure region properties
 rs_k = regionprops(l_)
 im_props = regionprops(l_, intensity_image=img_test)
 results = []

 # only append focus properties if single object is detected in magnitude spectrum
 # if there is an item in im_props, then for loop adds to properties each focus_blob's euler number and solidity value
 # if # of items in im_props is not 1, then skips to else
 if (len(im_props) == 1):
 for focus_blob in im_props:
 # if (focus_blob.area > 1000):
 # continue
 # blob = im_props[30]
 properties.append(focus_blob.euler_number)
 properties.append(focus_blob.solidity)
 stop = 1
 #drop removes dataframe index column and prints "focus dropped" message if # of items is not 1
 else:
 df_foci.drop(index, inplace=True)
 print('Focus bbox has more or less than 1 object. Focus dropped.')
 continue

 # F) FFT Magnitude properties
 # ft
 f = np.fft.fft2(img_test)
 # magnitude: ft
 magnitude_spectrum_ft = 20 * np.log(np.abs(f))
 # shift ft
 fshift = np.fft.fftshift(f)
 # magnitude: shifted ft
 magnitude_spectrum_fshift = 20 * np.log(np.abs(fshift))
 magnitude_spectrum_fshift_32F = magnitude_spectrum_fshift.astype(np.float32) # convert to uint8

 # F.A) Blur detection
 # F.A.0) Blur detection, Variance of Laplacian
 # save img as 32-bit with magnitude spectrum plot describing frequency and intensity (fast fourier transform)
 # different extensions to identify both locations where img w variance is saved, applying mag. shift
 fft_var_laplacian = cv2.Laplacian(magnitude_spectrum_fshift_32F, cv2.CV_32F).var() # variance of laplacian
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_5mag.tif', magnitude_spectrum_fshift_32F) # save img
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_5vlm.tif', magnitude_spectrum_fshift_32F) # save img

 # F.A.1) blur detection using measure function
 fft_blur_2 = measure.blur_effect(magnitude_spectrum_fshift_32F)

 # F.B) shannon entropy
 fft_shannon_entropy = measure.shannon_entropy(magnitude_spectrum_fshift_32F)

 # F.C) Stats
 fft_mean = np.mean(magnitude_spectrum_fshift_32F)
 fft_med = np.median(magnitude_spectrum_fshift_32F)
 fft_sum = np.sum(magnitude_spectrum_fshift_32F)
 fft_std = np.std(magnitude_spectrum_fshift_32F)
 fft_var = np.var(magnitude_spectrum_fshift_32F)
 fft_max = np.max(magnitude_spectrum_fshift_32F)
 fft_min = np.min(magnitude_spectrum_fshift_32F)

 # F.D) Other Geometric Properties
 # 1.1 OTSU thresholding
 # numpy isnan function checks if the input is a NaN (meaning not a number), then returns result as boolean array
 # any tests all array elements along all axis, bc an axis has not been specified in parameters
 # prints message and drops row if NaN is True
 if (np.isnan(magnitude_spectrum_fshift_32F).any() == True):
 print("Drop focus, contains nan in magnitude spectrum.")
 df_foci.drop(index, inplace=True)
 continue
 magnitude_spectrum_fshift_32F_gb = cv2.GaussianBlur(magnitude_spectrum_fshift_32F, (3, 3), 0)
 # save mag shift img with computer vision 2 gaussian blur applied to region (3,3) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_gb3_mag.tif', magnitude_spectrum_fshift_32F_gb)

 th = filters.threshold_otsu(magnitude_spectrum_fshift_32F_gb)
 # filters.threshold_otsu returns threshold value of img using otsu's method, and stores this value as th
 # greater than comparison returns true if mag shift is greater than th; this sets the threshold + stores it in img_mask
 img_mask = magnitude_spectrum_fshift_32F_gb > th
 # save img_masked image with mag shift/otsu filter applied to region (3,3) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_OTSUthresh_mag.tif', img_mask)

 # 1.2 connected component labeling
 # img_mask.reshape(n_rows, n_cols)
 # mh.label - labels contiguous regions of the image
 # saves img with labels; labelled images are integer images where values correspond to different regions
 l_, n_ = mh.label(img_mask, np.ones((3, 3), bool)) # binary_closed_hztl_k
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_CCL_mag.tif', l_)

 # 1.3 measure region properties
 rs_k = regionprops(l_)
 im_props = regionprops(l_, intensity_image=magnitude_spectrum_fshift_32F_gb)
 results = []

 # only append focus properties if single object is detected in magnitude spectrum
 #  if there is an item in im_props, then for loop adds to properties each focus_blob's euler number and solidity value
 #  if # of items in im_props is not 1, then skips to else
 if (len(im_props) == 1):
 for blob in im_props:
 # if (blob.area > 1000):
 # continue
 # blob = im_props[30]
 properties.append(blob.orientation)
 properties.append(blob.area)
 properties.append(blob.perimeter)
 properties.append(blob.major_axis_length)
 properties.append(blob.minor_axis_length)
 properties.append(blob.eccentricity)
 properties.append(blob.euler_number)
 properties.append(blob.solidity)

 fft_area = blob.area
 fft_perimeter = blob.perimeter
 # F.E) compactness
 # F.E.0) Polsby-Popper
 fft_compactness_pp = (4 * pi * fft_area) / (focus_perim * fft_perimeter)
 properties.append(fft_compactness_pp)
 # F.E.1) Schwartzberg
 fft_compactness_s = 1 / (fft_perimeter / (2 * pi * sqrt(fft_area / pi)))
 properties.append(fft_compactness_s)
 #drop removes dataframe index column and prints "focus dropped" message if # of items is not 1
 else:
 df_foci.drop(index, inplace=True)
 print('FFT magnitude has more or less than 1 object. Focus dropped.')
 continue

 # store first element in properties in area
# conditional statement that checks if the area of the foci is larger than 10000, if it is, then row is dropped + msg
 area = properties[1]
 if (area > 10000):
 df_foci.drop(index, inplace=True)
 print('FFT magnitude object area is too large. Focus dropped.')
 continue

 # Append rest of fft properties
 properties.append(fft_var_laplacian)
 properties.append(fft_blur_2)
 properties.append(fft_shannon_entropy)
 properties.append(fft_mean)
 properties.append(fft_med)
 properties.append(fft_sum)
 properties.append(fft_std)
 properties.append(fft_var)
 properties.append(fft_max)
 properties.append(fft_min)

 # Append rest of focus properties
 properties.append(var_laplacian)
 properties.append(var_laplacian_gb0)
 properties.append(var_laplacian_gb1)
 properties.append(var_laplacian_gb2)
 properties.append(var_laplacian_gb3)
 properties.append(blur_2)
 properties.append(shannon_entropy)
 properties.append(focus_mean)
 properties.append(focus_med)
 properties.append(focus_sum)
 properties.append(focus_std)
 properties.append(focus_var)
 properties.append(focus_max)
 properties.append(focus_min)
 properties.append(compactness_pp)
 properties.append(compactness_s)
 properties.append(len_img)

 # numpy isnan function checks if the input is a NaN (meaning not a number), then returns result as boolean array
 # any tests all array elements along all axis, bc an axis has not been specified in parameters
 # prints message and drops row if NaN is True, then adds properties to properties_full list
 if (np.isnan(properties).any() == True):
 print("Drop focus, contains nan in metrics.")
 df_foci.drop(index, inplace=True)
 continue
 properties_full.append(properties)

 # conditional statement: if the # of items in df_foci is less than 1, then prints message and numpy creates array of list
 # saves to new variable properties_full_ar
 if (len(df_foci) < 1):
 print('Focus dataframe empty. Skip this ROI.')
 continue
 properties_full_ar = np.array(properties_full)


 df_foci['euler_number'] = properties_full_ar[:, 0]
 df_foci['solidity'] = properties_full_ar[:, 1]

 df_foci['fft_orientation'] = properties_full_ar[:, 2]
 df_foci['fft_area'] = properties_full_ar[:, 3]
 df_foci['fft_perimeter'] = properties_full_ar[:, 4]
 df_foci['fft_major_axis_length'] = properties_full_ar[:, 5]
 df_foci['fft_minor_axis_length'] = properties_full_ar[:, 6]
 df_foci['fft_eccentricity'] = properties_full_ar[:, 7]
 df_foci['fft_euler_number'] = properties_full_ar[:, 8]
 df_foci['fft_solidity'] = properties_full_ar[:, 9]
 df_foci['fft_compactness_pp'] = properties_full_ar[:, 10]
 df_foci['fft_compactness_s'] = properties_full_ar[:, 11]
 df_foci['fft_var_laplacian'] = properties_full_ar[:, 12]
 df_foci['fft_blur_2'] = properties_full_ar[:, 13]
 df_foci['fft_shannon_entropy'] = properties_full_ar[:, 14]
 df_foci['fft_mean'] = properties_full_ar[:, 15]
 df_foci['fft_med'] = properties_full_ar[:, 16]
 df_foci['fft_sum'] = properties_full_ar[:, 17]
 df_foci['fft_std'] = properties_full_ar[:, 18]
 df_foci['fft_var'] = properties_full_ar[:, 19]
 df_foci['fft_max'] = properties_full_ar[:, 20]
 df_foci['fft_min'] = properties_full_ar[:, 21]

 df_foci['var_lap'] = properties_full_ar[:, 22]
 df_foci['var_lap_gb3'] = properties_full_ar[:, 23]
 df_foci['var_lap_gb5'] = properties_full_ar[:, 24]
 df_foci['var_lap_gb7'] = properties_full_ar[:, 25]
 df_foci['var_lap_gb9'] = properties_full_ar[:, 26]
 df_foci['blur_2'] = properties_full_ar[:, 27]
 df_foci['shannon_entropy'] = properties_full_ar[:, 28]
 df_foci['focus_mean'] = properties_full_ar[:, 29]
 df_foci['focus_med'] = properties_full_ar[:, 30]
 df_foci['focus_sum'] = properties_full_ar[:, 31]
 df_foci['focus_std'] = properties_full_ar[:, 32]
 df_foci['focus_var'] = properties_full_ar[:, 33]
 df_foci['focus_max'] = properties_full_ar[:, 34]
 df_foci['focus_min'] = properties_full_ar[:, 35]
 df_foci['compactness_pp'] = properties_full_ar[:, 36]
 df_foci['compactness_s'] = properties_full_ar[:, 37]
 df_foci['tot_pix_no'] = properties_full_ar[:, 38]

 df_foci.to_csv(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + 'focus_info_full_props.csv')

 stop = 1
 print('Next experiment...')

def measure_props_df(summary_dir, dir_list):
 all_files = []
 #dir_list= dir_list[12:14] # for debugging
 for dir_ in dir_list:
 #dir_ = dir_list[7] # for debugging
 # load all tiff movie files that end in _N_roi.tif (N == a number)
 files = glob(dir_[5] + '/*.roi')
 """iterating through public directories (what does dir_i? mean) and stores in some_files"""
 for dir_i in dir_list:
 some_files = glob(dir_i[5] + '/*.roi') # change to _roi.tif later
 """adds roi string, and if there are any files stored in some_files, add stored element to the list in all_files"""
 if (len(some_files) > 0):
 all_files.append(some_files)
 some_files = []
 else:
 continue

 # skip if there are no rois
 # if there are no files in all_files, then continue with constructing a dataframe and storing it to all_spools
 # then, apply pd.series to create columns in dataframe and stack restructures the multi-level columns to optimize rows
 # then, unique find the unique values in series; series meaning a single row
 # lspools stores series from spools after converting values into a list
 if (len(all_files) < 1):
 continue
 all_spools = pd.DataFrame(all_files)
 spools = all_spools.apply(pd.Series).stack().unique()
 lspools = spools.tolist()

 # obtain number of samples
 # looping through unique files (removing duplicate) and generate range returning list of #'s corresponding to file
 # for each loop num assumes a new value
 # then compile to regular expression and seperate from # and  the add sep_file's 3rd element to target file sep_files
 sep_files = []
 for num in range(0, len(lspools)):
 sep_file = re.compile(r"[\\]").split(lspools[num])
 sep_files.append(sep_file[2])
 sep_file = []

 counts, values = pd.Series(sep_files).value_counts().values, pd.Series(sep_files).value_counts().index
 df_results = pd.DataFrame(list(zip(values, counts)), columns=["value", "count"])
 sample_counts_dir = df_results.loc[df_results["value"] == dir_[3]]

 # Find ROI file
 # single ROI
 # count # of roi files in roi_list and if length is 1, then there isn't a zipped file
 # if the length is less than 1, then there are no files in this list, so it scans for a zipped file
 # for lengths beyond these conditions, it will print message and it will continue to the next iteration
 roi_list = glob(dir_[5] + '/*.roi')
 if (len(roi_list) == 1):
 roi_is_zip = False
 elif (len(roi_list) < 1):
 # more than one ROI
 roi_list = glob(dir_[5] + '/*.zip')
 roi_is_zip = True
 else:
 print("Missing ROI file for movie file")
 continue

 # scanning for roi_file in roi_list
 for roi_file in roi_list:
 # load the ROI and extract the coords
 # make a mask from the coords
 # create folder for this cropped out cell
 # use mask to save the cropped movie file
 if (roi_is_zip):
 roi_points = read_roi.read_roi_zip(roi_file)
 else:
 """if the zipped file DNE, then it will store the opened roi file that is read in binary mode as 
 fobj and then read fobj and store in roi_points, and then matches file formats as if it is a zip file"""
 fobj = open(roi_file, 'r+b')
 roi_points = read_roi.read_roi(fobj)
 roi_points = [roi_points, ] # make format same as if its a zip

 # ROI is bounding box, make mask
 # go through each frame and apply masks, save cropped movie file
 #roi_points_test = roi_points[1] # for debugging
 """based on file coords, roi_points = roi_count, bbox_points = second-coord, add # values to refer to file types"""
 for roi_count, bbox_points in enumerate(roi_points):#enumerate(roi_points_test):
 # roi_count = 1 # for debugging
 file = []
 # for loop is scanning for unique files with desired extensions for storing them in roifile
 for ext in ('/*.roi', '/*.zip'):
 roifile = glob(dir_[5] + ext)
 """if there are any files in roifile, joins the files as one iterable item and adds it to roifile, then prints message
 for each file"""
 if (len(roifile) > 0):
 file.append(''.join(roifile))
 file = ''.join(file)
 print('Processing ', file[:-4] + '_' + str(roi_count + 1))
 file_root = os.path.split(file)[1][:-4]
 sep_file_ = re.compile(r"[\\]").split(file)
 roi_folder = file_root + '_' + str(roi_count + 1)
 date_file = sep_file_[1][:-7]
 test = dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status_G.csv'
 # check for csv file with manually classified foci
 # if the file path exists then save read/retrieved data from .csv file to df_foci
 (dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status_G.csv') == True):
 df_foci = pd.read_csv(dir_[5] + '/' + roi_folder + '/' + 'focus_info_full_status_G.csv')
 else:
 continue

 #df_foci = pd.read_csv(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[4] + 'focus_info_full.csv')
 properties_full = []
 #df_foci_test = df_foci.iloc[26]
 #iterates through each row of the dataframe and prints message for each; stores value from row dictionary under
 # column "focus_label" to local focus_label variable
 for index, row in df_foci.iterrows():
 focus_label = row['focus_label']
 #print(index), 28
 #index = 70
 #row = df_foci.iloc[26]
 print('Focus label: ' + str(row['focus_label']))
 print('Index: ' + str(index))
 properties = []

 # stores in img read roi file as preprocessing + tif/high quality image; importing images, max proj, no outlines
 img = io.imread(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[4] + '_preproc_02_clahe_roi'+ str(roi_count + 1) + '_G.tif') # import max projection image

 # extract focus bounding box
 bbox = row['bbox'].split('(', 2)
 test1 = bbox[1].split(')', 2)
 test2 = test1[0].split(',', 4)
 results = list(map(int, test2))
 img_test = img[results[0]:results[2], results[1]:results[3]]
 img_test = (img_test - np.min(img_test)) / (np.max(img_test) - np.min(img_test)) # scale from 0 to 1
 img_test = np.multiply(img_test, 255) # scale from 0 to 255
 img_test = img_test.astype(np.float32)
 # pixel no
 len_img = img_test.size

 # save focus in focus folder
 # save images of individual foci, with and without outlines

 # A) BLUR DETECTION
 # A.0) blur detection with variance of laplacian
 # 0 mean proj single foci
 # stores laplacian transformed image with calculated variance to var_laplacian and saves img in roi file
 var_laplacian = cv2.Laplacian(img_test, cv2.CV_32F).var() # variance of laplacian
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_0vl_test.tif', img_test) # save img
 laplacian = cv2.Laplacian(img_test, cv2.CV_32F) # laplacian
 # apply laplacian operator to detect edges within img and save locally without variance
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_0l_test.tif', laplacian) # save img

 # 1 gaussian blur 0 of 0
 img_test_gb0 = cv2.GaussianBlur(img_test, (3, 3), 0)
 var_laplacian_gb0 = cv2.Laplacian(img_test_gb0, cv2.CV_32F).var() # variance of laplacian
 #save img from roifile with computer vision 2 gaussian blur + laplacian applied to region (3,3) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_1vlgb0_test.tif', img_test_gb0) # save img
 laplacian_gb0 = cv2.Laplacian(img_test_gb0, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb0 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_1l_gb0_test.tif', laplacian_gb0) # save img

 # 2 gaussian blur 1 of 0
 img_test_gb1 = cv2.GaussianBlur(img_test, (5, 5), 0)
 var_laplacian_gb1 = cv2.Laplacian(img_test_gb1, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (5,5) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_2vlgb1_test.tif', img_test_gb1) # save img
 laplacian_gb1 = cv2.Laplacian(img_test_gb1, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb1 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_2l_gb1_test.tif', laplacian_gb1) # save img


 # 3 gaussian blur 2 of 0
 img_test_gb2 = cv2.GaussianBlur(img_test, (7, 7), 0)
 var_laplacian_gb2 = cv2.Laplacian(img_test_gb2, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (7,7) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_3vlgb2_test.tif', img_test_gb2) # save img
 laplacian_gb2 = cv2.Laplacian(img_test_gb2, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb2 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_3l_gb2_test.tif', laplacian_gb2) # save img

 # 4 gaussian blur 3 of 0
 img_test_gb3 = cv2.GaussianBlur(img_test, (9, 9), 0)
 var_laplacian_gb3 = cv2.Laplacian(img_test_gb3, cv2.CV_32F).var() # variance of laplacian
 #save img with gaussian blur + laplacian applied to region (9,9) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_4vlgb3_test.tif', img_test_gb3) # save img
 laplacian_gb3 = cv2.Laplacian(img_test_gb3, cv2.CV_32F) # laplacian
 #save img with laplacian applied to 32-bit image, new variable compared to last (gb3 distinction)
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_4l_gb3_test.tif', laplacian_gb3) # save img

 # A.1) blur detection using measure function
 blur_2 = measure.blur_effect(img_test)

 # B) shannon entropy
 shannon_entropy = measure.shannon_entropy(img_test)

 # C) stats
 focus_mean = np.mean(img_test)
 focus_med = np.median(img_test)
 focus_sum = np.sum(img_test)
 focus_std = np.std(img_test)
 focus_var = np.var(img_test)
 focus_max = np.max(img_test)
 focus_min = np.min(img_test)

 # D) compactness
 focus_area = row['area']
 focus_perim = row['perimeter']

# checks for value of focus perimeter in dictionary and drops indices with perimeters less than 1; then, prints message
 if (focus_perim < 1):
 df_foci.drop(index, inplace=True)
 print('Focus is too small. Focus dropped.')
 continue

 # D.0) Polsby-Popper
 compactness_pp = (4 * pi * focus_area) / (focus_perim * focus_perim)

 # D.1) Schwartzberg
 compactness_s = 1 / (focus_perim / (2 * pi * sqrt(focus_area / pi)))

 # E) other geometric properties
 # 1.2 connected component labeling
 # img_mask.reshape(n_rows, n_cols)
 l_, n_ = mh.label(img_test, np.ones((3, 3), bool)) # binary_closed_hztl_k
 #saves the roi_file with string applied to extract row that contains focus_label???
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_CCL_mag.tif', l_)

 # 1.3 measure region properties
 rs_k = regionprops(l_)
 im_props = regionprops(l_, intensity_image=img_test)
 results = []

 # only append focus properties if single object is detected in magnitude spectrum
 # if there is an item in im_props, then for loop adds to properties each focus_blob's euler number and solidity value
 # if # of items in im_props is not 1, then skips to else
 if (len(im_props) == 1):
 for focus_blob in im_props:
 # if (focus_blob.area > 1000):
 # continue
 # blob = im_props[30]
 properties.append(focus_blob.euler_number)
 properties.append(focus_blob.solidity)
 stop = 1
 #drop removes dataframe index column and prints "focus dropped" message if a single object has not been detected
 else:
 df_foci.drop(index, inplace=True)
 print('Focus bbox has more or less than 1 object. Focus dropped.')
 continue

 # F) FFT Magnitude properties
 # ft
 f = np.fft.fft2(img_test)
 # magnitude: ft
 magnitude_spectrum_ft = 20 * np.log(np.abs(f))
 # shift ft
 fshift = np.fft.fftshift(f)
 # magnitude: shifted ft
 magnitude_spectrum_fshift = 20 * np.log(np.abs(fshift))
 magnitude_spectrum_fshift_32F = magnitude_spectrum_fshift.astype(np.float32) # convert to uint8

 # F.A) Blur detection
 # F.A.0) Blur detection, Variance of Laplacian
 # save img as 32-bit with magnitude spectrum plot describing frequency and intensity (fast fourier transform)
 # different extensions to identify both locations where img w variance is saved, applying mag. shift
 fft_var_laplacian = cv2.Laplacian(magnitude_spectrum_fshift_32F, cv2.CV_32F).var() # variance of laplacian
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_5mag.tif', magnitude_spectrum_fshift_32F) # save img
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_5vlm.tif', magnitude_spectrum_fshift_32F) # save img

 # F.A.1) blur detection using measure function
 fft_blur_2 = measure.blur_effect(magnitude_spectrum_fshift_32F)

 # F.B) shannon entropy
 fft_shannon_entropy = measure.shannon_entropy(magnitude_spectrum_fshift_32F)

 # F.C) Stats
 fft_mean = np.mean(magnitude_spectrum_fshift_32F)
 fft_med = np.median(magnitude_spectrum_fshift_32F)
 fft_sum = np.sum(magnitude_spectrum_fshift_32F)
 fft_std = np.std(magnitude_spectrum_fshift_32F)
 fft_var = np.var(magnitude_spectrum_fshift_32F)
 fft_max = np.max(magnitude_spectrum_fshift_32F)
 fft_min = np.min(magnitude_spectrum_fshift_32F)

 # F.D) Other Geometric Properties
 # 1.1 OTSU thresholding
 # numpy isnan function checks if the input is a NaN (meaning not a number), then returns result as boolean array
 # any tests all array elements along all axis, bc an axis has not been specified in parameters
 # prints message and drops row if NaN is True
 if (np.isnan(magnitude_spectrum_fshift_32F).any() == True):
 print("Drop focus, contains nan in magnitude spectrum.")
 df_foci.drop(index, inplace=True)
 continue
 magnitude_spectrum_fshift_32F_gb = cv2.GaussianBlur(magnitude_spectrum_fshift_32F, (3, 3), 0)
 # save mag shift img with computer vision 2 gaussian blur applied to region (3,3) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_gb3_mag.tif', magnitude_spectrum_fshift_32F_gb)

 th = filters.threshold_otsu(magnitude_spectrum_fshift_32F_gb)
 # filters.threshold_otsu returns threshold value of img using otsu's method, and stores this value as th
 # greater than comparison returns true if mag shift is greater than th; this sets the threshold + stores it in img_mask
 img_mask = magnitude_spectrum_fshift_32F_gb > th
 # save img_masked image with mag shift/otsu filter applied to region (3,3) -ksize
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_OTSUthresh_mag.tif', img_mask)

 # 1.2 connected component labeling
 #mh.label - labels contiguous regions of the image
 # img_mask.reshape(n_rows, n_cols)
 l_, n_ = mh.label(img_mask, np.ones((3, 3), bool)) # binary_closed_hztl_k
 # mh.label - labels contiguous regions of the image
 # saves img with labels; labelled images are integer images where values correspond to different regions
 io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
 4] + '_focusNo' + str(row['focus_label']) + '_roi' + str(roi_count + 1) + '_CCL_mag.tif', l_)

 # 1.3 measure region properties
 rs_k = regionprops(l_)
 im_props = regionprops(l_, intensity_image=magnitude_spectrum_fshift_32F_gb)
 results = []

 # only append focus properties if single object is detected in magnitude spectrum
 # if there is an item in im_props, then for loop adds to properties each focus_blob's euler number and solidity value
 # if # of items in im_props is not 1, then skips to else
 if (len(im_props) == 1):
 for blob in im_props:
 # if (blob.area > 1000):
 # continue
 # blob = im_props[30]
 properties.append(blob.orientation)
 properties.append(blob.area)
 properties.append(blob.perimeter)
 properties.append(blob.major_axis_length)
 properties.append(blob.minor_axis_length)
 properties.append(blob.eccentricity)
 properties.append(blob.euler_number)
 properties.append(blob.solidity)

 fft_area = blob.area
 fft_perimeter = blob.perimeter
 # F.E) compactness
 # F.E.0) Polsby-Popper
 fft_compactness_pp = (4 * pi * fft_area) / (focus_perim * fft_perimeter)
 properties.append(fft_compactness_pp)
 # F.E.1) Schwartzberg
 fft_compactness_s = 1 / (fft_perimeter / (2 * pi * sqrt(fft_area / pi)))
 properties.append(fft_compactness_s)
 #drop removes dataframe index column and prints "focus dropped" message if a single object has not been detected
 else:
 df_foci.drop(index, inplace=True)
 print('FFT magnitude has more or less than 1 object. Focus dropped.')
 continue

 # store first element in properties in area
 # conditional statement that checks if the area of the foci is larger than 1000, if it is, then row is dropped + msg
 area = properties[1]
 if (area > 1000):
 df_foci.drop(index, inplace=True)
 print('FFT magnitude object area is too large. Focus dropped.')
 continue

 # Append rest of fft properties
 properties.append(fft_var_laplacian)
 properties.append(fft_blur_2)
 properties.append(fft_shannon_entropy)
 properties.append(fft_mean)
 properties.append(fft_med)
 properties.append(fft_sum)
 properties.append(fft_std)
 properties.append(fft_var)
 properties.append(fft_max)
 properties.append(fft_min)

 # Append rest of focus properties
 properties.append(var_laplacian)
 properties.append(var_laplacian_gb0)
 properties.append(var_laplacian_gb1)
 properties.append(var_laplacian_gb2)
 properties.append(var_laplacian_gb3)
 properties.append(blur_2)
 properties.append(shannon_entropy)
 properties.append(focus_mean)
 properties.append(focus_med)
 properties.append(focus_sum)
 properties.append(focus_std)
 properties.append(focus_var)
 properties.append(focus_max)
 properties.append(focus_min)
 properties.append(compactness_pp)
 properties.append(compactness_s)
 properties.append(len_img)

 # numpy isnan function checks if the input is a NaN (meaning not a number), then returns result as boolean array
 # any tests all array elements along all axis, bc an axis has not been specified in parameters
 # prints message and drops row if NaN is True, then adds properties to properties_full list
 if (np.isnan(properties).any() == True):
 print("Drop focus, contains nan in metrics.")
 df_foci.drop(index, inplace=True)
 continue
 properties_full.append(properties)

 # conditional statement: if the # of items in df_foci is less than 1, then prints message and numpy creates array of list
 # saves to new variable properties_full_ar
 if (len(df_foci) < 1):
 print('Focus dataframe empty. Skip this ROI.')
 continue
 properties_full_ar = np.array(properties_full)


 df_foci['euler_number'] = properties_full_ar[:, 0]
 df_foci['solidity'] = properties_full_ar[:, 1]

 df_foci['fft_orientation'] = properties_full_ar[:, 2]
 df_foci['fft_area'] = properties_full_ar[:, 3]
 df_foci['fft_perimeter'] = properties_full_ar[:, 4]
 df_foci['fft_major_axis_length'] = properties_full_ar[:, 5]
 df_foci['fft_minor_axis_length'] = properties_full_ar[:, 6]
 df_foci['fft_eccentricity'] = properties_full_ar[:, 7]
 df_foci['fft_euler_number'] = properties_full_ar[:, 8]
 df_foci['fft_solidity'] = properties_full_ar[:, 9]
 df_foci['fft_compactness_pp'] = properties_full_ar[:, 10]
 df_foci['fft_compactness_s'] = properties_full_ar[:, 11]
 df_foci['fft_var_laplacian'] = properties_full_ar[:, 12]
 df_foci['fft_blur_2'] = properties_full_ar[:, 13]
 df_foci['fft_shannon_entropy'] = properties_full_ar[:, 14]
 df_foci['fft_mean'] = properties_full_ar[:, 15]
 df_foci['fft_med'] = properties_full_ar[:, 16]
 df_foci['fft_sum'] = properties_full_ar[:, 17]
 df_foci['fft_std'] = properties_full_ar[:, 18]
 df_foci['fft_var'] = properties_full_ar[:, 19]
 df_foci['fft_max'] = properties_full_ar[:, 20]
 df_foci['fft_min'] = properties_full_ar[:, 21]

 df_foci['var_lap'] = properties_full_ar[:, 22]
 df_foci['var_lap_gb3'] = properties_full_ar[:, 23]
 df_foci['var_lap_gb5'] = properties_full_ar[:, 24]
 df_foci['var_lap_gb7'] = properties_full_ar[:, 25]
 df_foci['var_lap_gb9'] = properties_full_ar[:, 26]
 df_foci['blur_2'] = properties_full_ar[:, 27]
 df_foci['shannon_entropy'] = properties_full_ar[:, 28]
 df_foci['focus_mean'] = properties_full_ar[:, 29]
 df_foci['focus_med'] = properties_full_ar[:, 30]
 df_foci['focus_sum'] = properties_full_ar[:, 31]
 df_foci['focus_std'] = properties_full_ar[:, 32]
 df_foci['focus_var'] = properties_full_ar[:, 33]
 df_foci['focus_max'] = properties_full_ar[:, 34]
 df_foci['focus_min'] = properties_full_ar[:, 35]
 df_foci['compactness_pp'] = properties_full_ar[:, 36]
 df_foci['compactness_s'] = properties_full_ar[:, 37]
 df_foci['tot_pix_no'] = properties_full_ar[:, 38]

 df_foci.to_csv(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + 'focus_info_full_props.csv')

 stop = 1
 print('Next experiment...')

 elif (func == 'measure_props'):
 dirs_dict = get_dir_list(base_dir)
 for summary_dir in dirs_dict.keys():
 # summary_dir = '2022_idr_data'
 measure_props(base_dir + '/' + summary_dir, dirs_dict[summary_dir])
 print('Done.')

elif (func == 'measure_props_idr'):
dirs_dict = get_dir_list(base_dir)
for summary_dir in dirs_dict.keys():
    summary_dir = '2022_idr_data'
measure_props_idr(base_dir + '/' + summary_dir, dirs_dict[summary_dir])
print('Done.')

elif (func == 'measure_props_df'):
dirs_dict = get_dir_list(base_dir)
for summary_dir in dirs_dict.keys():
    summary_dir = '2022_DF_data'
measure_props_df(base_dir + '/' + summary_dir, dirs_dict[summary_dir])
print('Done.')
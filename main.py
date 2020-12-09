from datetime import datetime
from SeamCarving.utilities import *
from SeamCarving.parameters import *

# Eneregy Function, Convert the image to Gray scale and apply the sobel filter
def compute_energy_function(img):
    timg = deprocess_image(img)
    ddepth = cv2.CV_16S
    gray = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0,)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad_mag = cv2.add(abs_grad_x, abs_grad_y)
    grad_mag = grad_mag.astype(np.float64)
    return grad_mag

# Cumulative Eneregy Function
def compute_cumulative_minimum_energy(energy) :
    M = energy.copy()
    M_path = M.copy()
    M_path.fill(0)

    # M(i, j) = e(i, j) + min(M(i−1, j−1), M(i−1, j), M(i−1, j + 1))
    for i in range (1,M.shape[0]) :
        for j in range (0, M.shape[1]) :
            start_j = 0 if (j - 1 ) < 0 else j-1
            end_j = M.shape[1] if j+2 >= M.shape[1] else j + 2

            min_j = start_j + np.argmin(M[i-1, start_j : end_j])
            M[i,j] += M[i-1, min_j ]
            M_path[i, j] = min_j

    return M, M_path

def find_optimal_seam(M) :
    rows,cols = M.shape
    path = np.zeros(rows, dtype= np.int32)

# starting point
    min_j = int(np.argmin(M[rows-1]))
    path[rows-1] = min_j

# backtrack through index map
    for i in range (rows-2, -1, -1) :
        start_j = 0 if (min_j - 1) < 0 else min_j - 1
        end_j = cols if min_j + 2 >= cols else min_j + 2

        min_j =  start_j + int(np.argmin(M[i, start_j : end_j] ))
        path[i] = min_j

    return path

def display_carve_seam_v2(img, path):
    rows, cols, _ = img.shape
    output = img.copy()

    for i in range(0, rows) :
        min_j = path[i]
        output[i, min_j] = RED_COLOR

    return output

def process_carve_seam_v2(img, path ,r_offset, mask, mode='REGULAR'):
    rows, cols, _ = img.shape
    new_cols = cols + r_offset

    output = np.zeros((rows, new_cols,3 ))
    new_mask = np.zeros((rows, new_cols,3 ))

    for i in range(0, rows) :
        min_j = path[i]
        for j in range (0, new_cols) :
            if r_offset < 0 :
                if j < min_j :
                    output[i,j] = img[i,j]
                    new_mask[i,j] = mask[i,j] if mask != [] else 0
                elif j >= min_j :
                    output[i,j] = img[i,j+1]
                    new_mask[i, j] = mask[i, j+1] if mask != [] else 0
            else :
                if j < min_j :
                    output[i,j] = img[i,j]
                elif j > min_j :
                    output[i,j] = img[i,j-1]
                else :
                    if mode == 'RED' :
                        output[i, j] = RED_COLOR
                    elif mode == 'SMARTINSERT' :
                        output[i, j] = MASK_COLOR
                    else :
                        output[i, j] = (img[i, j - 1] * 0.5 + img[i, j] * 0.5)

    return output, new_mask

def process_cols_seam(img, chg_x) :
    output = img.copy()
    step = int(chg_x / abs(chg_x)) if chg_x != 0 else 0
    Ms, paths = [] , []

    for x in range (0, abs(chg_x)):
        energy = compute_energy_function(output)
        M, _ = compute_cumulative_minimum_energy(energy)
        path = find_optimal_seam(M)
        paths.append(path)

        if VISUALIZE_FLAG == True :
            visual_map = display_carve_seam_v2(output,path)
            visual_map = deprocess_image(visual_map)
            cv2.imshow('Seam Carve', visual_map)
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            fname = 'Output/' + current_time + '.png'
            cv2.imwrite(fname, visual_map)
            cv2.waitKey(1)

        output, _ = process_carve_seam_v2(output, path ,step, [] , 'REGULAR')

    v_map = output.copy()
    if RECONSTRUCT_RED_LINE  == True:
        for i in range (len(paths)-1, -1, -1 ):
            wa_path = paths[i]
            v_map,_ = process_carve_seam_v2(v_map, wa_path, 1, [], 'RED')
            visual_map = deprocess_image(v_map)
            cv2.imshow('Red Lines', visual_map)
            cv2.waitKey(1)

        v_map = deprocess_image(v_map)
        cv2.imshow('Red Lines', v_map)

    return output


def process_rows_seam(img ,chg_y ) :
    output = img.copy()
    step = int(chg_y / abs(chg_y)) if chg_y != 0 else 0
    Ms, paths = [] , []

    output = rotate_image(output, cv2.ROTATE_90_CLOCKWISE)

    for x in range (0, abs(chg_y)):
        energy = compute_energy_function(output)
        M, _ = compute_cumulative_minimum_energy(energy)
        path = find_optimal_seam(M)
        paths.append(path)

        if VISUALIZE_FLAG == True:
            visual_map = display_carve_seam_v2(output,path)
            visual_map = rotate_image(visual_map, cv2.ROTATE_90_COUNTERCLOCKWISE )
            visual_map = deprocess_image(visual_map)
            cv2.imshow('Seam Carve', visual_map)
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            fname = 'Output/' + current_time + '.png'
            cv2.imwrite(fname, visual_map)
            cv2.waitKey(1)

        output, _ = process_carve_seam_v2(output, path, step, [] , 'REGULAR')

    v_map = output.copy()
    if RECONSTRUCT_RED_LINE  == True:
        for i in range (len(paths)-1, -1, -1 ):
            wa_path = paths[i]
            v_map, _ = process_carve_seam_v2(v_map, wa_path, 1, [] , 'RED')
            visual_map = deprocess_image(v_map)
            cv2.imshow('Red Lines', visual_map)
            cv2.waitKey(1)

        v_map = deprocess_image(v_map)
        cv2.imshow('Red Lines', v_map)

    output = rotate_image(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
    v_map = rotate_image(v_map, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return output

def test_energy_map(img) :
    energy = compute_energy_function(img)
    energy = deprocess_image(energy)
    cv2.imshow('energy',energy)
    cv2.waitKey(IM_SHOW_TIME)

def remove_seam(img,chg_x,chg_y) :
    assert chg_x <= 0, "remove seam has to input minus dimension chg_x parameter"
    assert chg_y <= 0, "remove seam has to input minus dimension chg_y parameter"

    output = process_cols_seam(img, chg_x)
    output = process_rows_seam(output,chg_y)
    #output = process_rows_seam(img, chg_y)
    return output

def insert_seam(img,chg_x,chg_y) :
    assert chg_x >= 0, "insert seam has to input postive dimension chg_x parameter"
    assert chg_y >= 0, "insert seam has to input postive dimension chg_y parameter"

    output = process_cols_seam(img ,chg_x)
    output = process_rows_seam(output, chg_y)
    return output

def smart_insert_process_cols_seam(img, chg_x) :
    rows, cols, _ = img.shape
    output = img.copy()
    Ms, paths = [] , []

    # compute paths and maps
    # collect  remove seams
    for x in range (0, abs(chg_x)):
        energy = compute_energy_function(output)
        M, _ = compute_cumulative_minimum_energy(energy)
        path = find_optimal_seam(M)
        paths.append(path)
        output, _ = process_carve_seam_v2(output, path , -1 , [] , 'REGULAR')

    v_map = output.copy()
    mask_map = output.copy()
    if RECONSTRUCT_RED_LINE  == True:
        for i in range (len(paths)-1, -1, -1 ):
            wa_path = paths[i]
            v_map, _ = process_carve_seam_v2(v_map, wa_path, 1, [], 'RED')
            mask_map, _ = process_carve_seam_v2(mask_map, wa_path, 1, [], 'SMARTINSERT')
            visual_map = deprocess_image(v_map)
            cv2.imshow('Red Lines', visual_map)
            cv2.waitKey(5)

        v_map = deprocess_image(v_map)
        cv2.imshow('Red Lines', v_map)
        cv2.waitKey(10000)
    # Insert Seam
    new_output = np.zeros( (rows, cols+chg_x, 3), dtype=np.float)

    for i in range (0, rows) :
        new_j = 0
        for j in range (0, cols):
            if (mask_map[i,j] != MASK_COLOR).all() :
                new_output[i, new_j] = img[i,j]
            else :
                new_output[i, new_j] = img[i, j]
                new_j = new_j + 1
                lj = 0 if (j-1) <0 else j-1
                rj = cols-1 if (j+1) > (cols-1) else j+1
                new_output[i,new_j] = img[i,lj] * 0.5 + img[i, rj] * 0.5

            new_j = new_j + 1
    return new_output

def smart_insert_process_rows_seam(img, chg_y) :
    t_img = rotate_image(img, cv2.ROTATE_90_CLOCKWISE)
    rows, cols, _ = img.shape
    output = t_img.copy()
    Ms, paths = [] , []

    # compute paths and maps
    # collect  remove seams
    for x in range(0, abs(chg_y)):
        energy = compute_energy_function(output)
        M, _ = compute_cumulative_minimum_energy(energy)
        path = find_optimal_seam(M)
        paths.append(path)
        output, _ = process_carve_seam_v2(output, path, -1, [], 'REGULAR')

    v_map = output.copy()
    mask_map = output.copy()
    if RECONSTRUCT_RED_LINE  == True:
        for i in range (len(paths)-1, -1, -1 ):
            wa_path = paths[i]
            v_map, _ = process_carve_seam_v2(v_map, wa_path, 1, [], 'RED')
            mask_map, _ = process_carve_seam_v2(mask_map, wa_path, 1, [], 'SMARTINSERT')
            visual_map = deprocess_image(v_map)
            cv2.imshow('Red Lines', visual_map)
            cv2.waitKey(5)

        v_map = deprocess_image(v_map)
        cv2.imshow('Red Lines', v_map)

    # Insert Seam
    new_output = np.zeros((cols, rows + chg_y, 3), dtype=np.float)
    for i in range(0, cols):
        new_j = 0
        for j in range(0, rows):
            if (mask_map[i, j] != MASK_COLOR).all():
                new_output[i, new_j] = t_img[i, j]
            else:
                new_output[i, new_j] = t_img[i, j]
                new_j = new_j + 1
                lj = 0 if (j - 1) < 0 else j - 1
                rj = rows - 1 if (j + 1) > (rows - 1) else j + 1
                new_output[i, new_j] = t_img[i, lj] * 0.5 + t_img[i, rj] * 0.5
            new_j = new_j + 1

    new_output = rotate_image(new_output, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return new_output

def smart_insert_seam(img,chg_x,chg_y) :
    assert chg_x >= 0, "insert seam has to input postive dimension chg_x parameter"
    assert chg_y >= 0, "insert seam has to input postive dimension chg_y parameter"

    output = smart_insert_process_cols_seam(img, chg_x)
    output = smart_insert_process_rows_seam(output,chg_y)
    #output = smart_insert_process_rows_seam(img, chg_y)
    return output

def amplify_content(img,ratio) :
    width = int(img.shape[1] * ratio / 100)
    height = int(img.shape[0] * ratio / 100)
    dim = (width, height)
    img2 = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    w_diff = img2.shape[1] - img.shape[1]
    h_diff = img2.shape[0] - img.shape[0]

    output = remove_seam(img2,w_diff*-1,h_diff*-1)
    return output

def multiply_mask (energy, mask) :
    cols,rows = energy.shape

    for i in range(0 , cols):
        for j in range(0, rows):
            if (mask[i,j] == WHITE_COLOR).all():
                energy[i,j] = energy[i,j] * (-99999999)

    return energy

def length_of_mask(mask) :
    cols,rows,_ = mask.shape
    len = 0
    for i in range(0 , cols):
        for j in range(0, rows):
            if (mask[i,j] == WHITE_COLOR).all():
                len = len + 1

    return len


def process_cols_seam_object_removal(img, mask) :
    output = img.copy()
    step = -1
    Ms, paths = [] , []


    while length_of_mask(mask) != 0:

        energy = compute_energy_function(output)
        energy = multiply_mask(energy, mask)
        M, _ = compute_cumulative_minimum_energy(energy)
        path = find_optimal_seam(M)
        paths.append(path)

        if VISUALIZE_FLAG == True :
            visual_map = display_carve_seam_v2(output,path)
            visual_map = deprocess_image(visual_map)
            cv2.imshow('Seam Carve', visual_map)
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            fname = 'Output/' + current_time + '.png'
            cv2.imwrite(fname, visual_map)
            cv2.waitKey(1)

        output, mask = process_carve_seam_v2(output, path ,step, mask, mode = 'REGULAR')

    v_map = output.copy()
    if RECONSTRUCT_RED_LINE  == True:
        for i in range (len(paths)-1, -1, -1 ):
            wa_path = paths[i]
            v_map,_ = process_carve_seam_v2(v_map, wa_path, 1, [], mode = 'RED')
            visual_map = deprocess_image(v_map)
            cv2.imshow('Red Lines', visual_map)
            cv2.waitKey(1)

        v_map = deprocess_image(v_map)
        cv2.imshow('Red Lines', v_map)

    return output

def remove_object(img, mask) :
    output = process_cols_seam_object_removal(img, mask)

    return output

def run() :

    img = load_image('Input/Penguin.png')
    img = img.astype(np.float64)

    # Test Energy Map
    #test_energy_map(img)

    # Revmoe Seam
    # output = remove_seam(img,-50,-50)

    # Naive Enlage
    #output = insert_seam(img,30,0)

    # Smart Enlarge
    #output = smart_insert_seam(img,30,30)

    # content Amplifier
    #output = amplify_content(img, ratio)

    # remove object
    mask = load_image('Input/Penguin_Mask.png')
    output = remove_object(img, mask)

    output = deprocess_image(output)
    img = deprocess_image(img)

    cv2.imshow('final',output)
    cv2.imshow('input',img)
    cv2.waitKey(IM_SHOW_TIME)
    cv2.imwrite('Output/output.png', output)

run()



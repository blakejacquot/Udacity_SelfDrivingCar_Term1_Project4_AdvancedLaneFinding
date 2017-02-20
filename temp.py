def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Args:
      img: Color image on to draw lines.
      lines: Lines from Hough transform.
      color: Color of superimposed lines.
      thickness: Thickness of lines (in pixels?).

    Returns:
      TBD

    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below

    """
    line_image = np.copy(img)*0 # Blank image on which to draw lines.
    shape_img = img.shape
    x_max = shape_img[1]
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns
      line_img: Image with hough lines drawn.
      lines: Hough lines from the transform of form x1,y1,x2,y2.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    Args:
      img: Output of the hough_lines(), An image with lines drawn on it.
           Should be a blank image (all black) with lines drawn on it.
      initial_img: image before any processing.
      α: TBD
      β: TBD
      λ: TBD

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def lane_mark_image(initial_image):
    """ Process initial image to include long lane markers, each composed of a single line.
    Args:
      initial_image: 3D numpy array

    Returns:
      final_image: 3D numpy array with lanes marked

    """
    # Grayscale the image.
    gray_image = grayscale(initial_image)

    # Gaussian blur the image
    kernel_size = 5
    blur_gray = gaussian_blur(gray_image, kernel_size)

    # Canny edge detection
    low_threshold = 50
    high_threshold = 150
    canny_blur_gray = canny(blur_gray, low_threshold, high_threshold)

    # Define four-sided polygon to mask image
    imshape = initial_image.shape
    vert1 = (130,imshape[0]) # bottom left
    vert2 = (440, 330) # top left
    vert3 = (525,330) # top right
    vert4 = (950, imshape[0]) # bottom right
    vertices = np.array([[vert1, vert2, vert3, vert4]], dtype=np.int32)
    masked_canny_blur_gray = region_of_interest(canny_blur_gray, vertices)

    # Hough transform to get all lines in the ROI
    rho = 3 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20 #minimum number of pixels making up a line
    max_line_gap = 50 # maximum gap in pixels between connectable line segments
    line_image = np.copy(initial_image)*0 # creating a blank to draw lines on
    lines = hough_lines(masked_canny_blur_gray, rho,
                                           theta, threshold, min_line_len, max_line_gap)

    # Set boundary values for drawing lines on final image
    y_min = 350 #y1, upper portion (apex of trapezoid)
    y_max = imshape[0] #y2, lower portion (lower frame boundary)
    x_max = imshape[1]-1

    out_lines = compute_long_line(lines, y_min, y_max, x_max)

    # Get dark image with red lines drawn on lane markers
    line_image = draw_lines(initial_image, out_lines, color=[255, 0, 0], thickness=10)

    # Superimpose red lane markers on original image
    final_img = weighted_img(line_image, initial_image, α=0.8, β=1., λ=0.)

    return final_img

def compute_long_line(lines, y_min, y_max, x_max):
    """Compute single line for left, right lanes
    Args:
      lines: Hough lines from the transform of form x1,y1,x2,y2.
      y_min: self explanatory
      y_max: self explanatory
      x_max: self explanatory

    Returns:
      lines_updated: ND array with longest line
    """
    neg_slope_longest_line = [0,0,0] # [m, b, line_length]
    pos_slope_longest_line = [0,0,0] # [m, b, line_length]
    neg_m_accumulator = 0
    neg_b_accumulator = 0
    neg_line_len_accumulator = 0
    pos_m_accumulator = 0
    pos_b_accumulator = 0
    pos_line_len_accumulator = 0

    for curr_line in lines:
        x1 = curr_line[0][0]
        y1 = curr_line[0][1]
        x2 = curr_line[0][2]
        y2 = curr_line[0][3]
        m = (y2-y1)/(x2-x1)
        b = y1 - m * x1
        triang_length = x2 - x1
        triang_height = y2 - y1
        line_len = math.sqrt(triang_length*triang_length + triang_height*triang_height)
        if m < 0: #negative slope
            neg_m_accumulator += m * line_len # weighted accumulator
            neg_b_accumulator += b * line_len # weighted accumulator
            neg_line_len_accumulator += line_len
        else: #positive slope
            pos_m_accumulator += m * line_len # weighted accumulator
            pos_b_accumulator += b * line_len # weighted accumulator
            pos_line_len_accumulator += line_len
    neg_m = neg_m_accumulator / neg_line_len_accumulator
    neg_b = neg_b_accumulator / neg_line_len_accumulator
    pos_m = pos_m_accumulator / pos_line_len_accumulator
    pos_b = pos_b_accumulator / pos_line_len_accumulator

    # Define x vals for these y vals
    x1_negslope = (y_min - neg_b) / neg_m
    x2_negslope = (y_max - neg_b) / neg_m
    x1_posslope = (y_min - pos_b) / pos_m
    x2_posslope = (y_max - pos_b) / pos_m

    if x1_negslope >= x_max:
        x1_negslope = x_max
    if x1_negslope < 0:
        x1_negslope = 0
    if math.isnan(x1_negslope):
        x1_negslope = 0


    if x2_negslope >= x_max:
        x2_negslope = x_max
    if x2_negslope < 0:
        x2_negslope = 0
    if math.isnan(x2_negslope):
        x2_negslope = 0


    if x1_posslope >= x_max:
        x1_posslope = x_max
    if x1_posslope < 0:
        x1_posslope = 0
    if math.isnan(x1_posslope):
        x1_posslope = 0

    if x2_posslope >= x_max:
        x2_posslope = x_max
    if x2_posslope < 0:
        x2_posslope = 0
    if math.isnan(x2_posslope):
        x2_posslope = 0

    x1_negslope = int(round(x1_negslope))
    x2_negslope = int(round(x2_negslope))
    x1_posslope = int(round(x1_posslope))
    x2_posslope = int(round(x2_posslope))
    neg_line = [x1_negslope, y_min, x2_negslope, y_max]
    pos_line = [x1_posslope, y_min, x2_posslope, y_max]
    out_lines = np.zeros((2, 1, 4), dtype=np.uint32)
    neg_line = np.array([neg_line[0], neg_line[1], neg_line[2], neg_line[3]])
    pos_line = np.array([pos_line[0], pos_line[1], pos_line[2], pos_line[3]])
    out_lines[0,0,:] = neg_line
    out_lines[1,0,:] = pos_line

    return out_lines

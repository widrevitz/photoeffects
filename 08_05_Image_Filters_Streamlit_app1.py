import io
import base64
import cv2
from PIL import Image
from filters import *


# Generating a link to download a particular image file.
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# Set title.
st.title("""Let's play with Photoshop version minus 11, also known as:  Let's play with Linear Algebra under the hood.
Code n' stuff freely 'borrowed' and tons of help from good ol' CHATgpt too""")
st.write("""==>This is a simple demo of two or three newish functions wee lil' Baba has been able to pick up in 
the last couple weeks.    The big pieces include using the Python programming language with certain helper libraries.

""")
st.write("""!!!!Up to you...skip the explanation and try right away if you want. !!!!   """)   
st.write("""     """)   
st.write("""The main operational pieces include 1) various ways to manipulate 
images to get all kinds of effects, and 2) a method ... quite notable these days... to 'publish the program onto a 
cloud server which can then become a web server.    The somewhat extensive program's functions then are run on the cloud
server.   And my own 'widdle RPi' is used therefore 'only' as a local, home development machine.   More later.    
Another web page, located at babaaifaceID.streamlit.app shows some quite different functinoality.   """)

st.write("""[Click here to visit babaaifaceID(https://babaaifaceID.streamlit.app)""")                       


st.write("""    """)
st.write(""" ==>Directions for use:   First add or drop a file in the space indicated on this page.   It should then
be displayed.    Then you can see what happens when different effects/operations happen to 'magically' affect how the 
image looks.   These are done by clicking on any desired effect as shown in examples under the image, then clicking on
the tiny square at the right top corner of the sample effect image.   After a second or two, and maybe clicking on
some blank space.....your own image will be processed in the way the sample image chosen had been processed.   Play with
the slider thing which pops up only if you pick an applicable effec.  No guarantees on that.  In fact...no guarantees at all!

Select a filter works best though.   Unhuman factors were used to put this thing together...We are 'just playing' 
at this after all.   Ha, ha, when in doubt, poke around!   Just like the  real apps.


""")

st.write("""   """)
st.write("""NO, Virginia, this isn't magic.   Magic will be summoned in the other website mentioned above.   This is
al done via mechanisms really well understood by even half mortals such as Ben, with a bit of study.  In other words,
no AI per se.   Maybe 'smart programs' but....no AI.    High level explanations will come one fine day in an email to
the unlucky few who get these, like them or not, read them or not.   """)


# Upload image.
uploaded_file = st.file_uploader("Choose an image file:", type=["png", "jpg"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    input_col, output_col = st.columns(2)
    with input_col:
        st.header("Original")
        # Display uploaded image.
        st.image(img, channels="BGR", use_container_width=True)

    st.header("Filter Examples:")
    # Display a selection box for choosing the filter to apply.
    option = st.selectbox(
        "Select a filter:",
        (
            "None",
            "Black and White",
            "Sepia / Vintage",
            "Vignette Effect",
            "Pencil Sketch",
        ),
    )

    # Define columns for thumbnail images.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Black and White")
        st.image("filter_bw.jpg")
    with col2:
        st.caption("Sepia / Vintage")
        st.image("filter_sepia.jpg")
    with col3:
        st.caption("Vignette Effect")
        st.image("filter_vignette.jpg")
    with col4:
        st.caption("Pencil Sketch")
        st.image("filter_pencil_sketch.jpg")

    # Flag for showing output image.
    output_flag = 1
    # Colorspace of output image.
    color = "BGR"

    # Generate filtered image based on the selected option.
    if option == "None":
        # Don't show output image.
        output_flag = 0
    elif option == "Black and White":
        output = bw_filter(img)
        color = "GRAY"
    elif option == "Sepia / Vintage":
        output = sepia(img)
    elif option == "Vignette Effect":
        level = st.slider("level", 0, 5, 2)
        output = vignette(img, level)
    elif option == "Pencil Sketch":
        ksize = st.slider("Blur kernel size", 1, 11, 5, step=2)
        output = pencil_sketch(img, ksize)
        color = "GRAY"

    with output_col:
        if output_flag == 1:
            st.header("Output")
            st.image(output, channels=color)
            # fromarray convert cv2 image into PIL format for saving it using download link.
            if color == "BGR":
                result = Image.fromarray(output[:, :, ::-1])
            else:
                result = Image.fromarray(output)
            # Display link.
            st.markdown(get_image_download_link(result, "output.png", "Download " + "Output"), unsafe_allow_html=True)

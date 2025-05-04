from tkinter import *
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
import numpy as np
from final_rgn1 import rgn_grwing  # Make sure to use the updated function name
from final_scores import calculate_jaccard, calculate_dice, calculate_hausdorff, calculate_precision_recall, calculate_sensitivity_specificity

window = Tk()
window.title('Region Growing GUI')
window.geometry('650x850')

# Global variables
seed = None
img = None
img_array = None
canvas_img = None
result_img = None
result_binary = None  # Store binary result directly
label_img_file = 'output.jpg'
label_binary = None  # Store binary label directly

# Try to load label image once at startup and invert black/white
try:
    label_img_pil = Image.open(label_img_file).resize((500, 500))
    label_array = np.array(label_img_pil.convert('L'))
    # Invert the black and white colors
    label_array = 255 - label_array  # Invert the grayscale values
    label_binary = label_array > 127  # Convert to binary once

    # Create display image from inverted array
    inverted_label_img = Image.fromarray(label_array)
    label_img = ImageTk.PhotoImage(inverted_label_img)  # Only for display
except Exception as e:
    print(f"Error loading label image: {e}")
    label_img_pil = None
    label_img = None
    label_binary = None

# Create canvas
canvas = Canvas(window, width=500, height=500, bg='white')
canvas.pack()

# Create widgets
select_btn = Button(window, text='Select Image File', command=lambda: select_img())
select_btn.pack()

coord_label = Label(window, text='Seed: Not Selected')
coord_label.pack()

proceed_btn = Button(window, text='Proceed', state='disabled', command=lambda: start_growing())
proceed_btn.pack()

vis_jac = Button(window, text='Visualize Jaccard Result', command=lambda: vis_jac())
vis_jac.pack()

error_label = Label(window, text='', fg='red')
error_label.pack()

t_scroll = Scale(window, from_=1, to=300, orient=HORIZONTAL, label='Threshold',
                 command=lambda T: update_region(T))
t_scroll.set(50)
t_scroll.pack(expand=1, fill=BOTH)

calc_scores_btn = Button(window, text='Calculate All Scores', command=lambda: calculate_all_scores())
calc_scores_btn.pack()


def select_img():
    global img, img_array, seed, canvas_img
    try:
        path = filedialog.askopenfilename(filetypes=[('Image Files', '*.png *.jpeg *.jpg')])
        if path:
            img = Image.open(path).resize((500, 500))
            img_array = np.array(img)  # Convert to numpy array once

            # Display the image
            canvas_img = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=NW, image=canvas_img)

            # Reset seed and UI state
            seed = None
            result_img = None
            result_binary = None
            coord_label.config(text='Seed: Not Selected')
            proceed_btn.config(state='disabled')
            error_label.config(text='')
    except Exception as e:
        error_label.config(text=f"Error loading image: {str(e)}")


def get_seed(event):
    global seed
    try:
        if img_array is None:
            error_label.config(text="Please select an image first")
            return

        # Get clicked coordinates
        seed = (event.y, event.x)

        # Ensure the coordinates are within bounds
        height, width = img_array.shape[0], img_array.shape[1]
        if 0 <= seed[0] < height and 0 <= seed[1] < width:
            coord_label.config(text=f'Seed: {seed}')
            proceed_btn.config(state='normal')
            error_label.config(text='')
        else:
            error_label.config(text="Seed point out of bounds")
            seed = None
    except Exception as e:
        error_label.config(text=f"Error selecting seed: {str(e)}")
        seed = None


def start_growing():
    try:
        if seed is None:
            error_label.config(text="Please select a seed point")
            return

        # Use the current threshold value
        threshold = t_scroll.get()
        update_region(threshold)
    except Exception as e:
        error_label.config(text=f"Error starting region growing: {str(e)}")


def update_region(T):
    global canvas_img, result_img, result_binary
    try:
        if seed is None or img_array is None:
            return

        # Convert to integer
        threshold = int(T)

        # Apply region growing
        result_img = rgn_grwing(seed, threshold, img_array)

        # Store binary version directly to avoid repeated conversions
        result_array = np.array(result_img)
        result_binary = result_array > 127

        # Display result
        canvas_img = ImageTk.PhotoImage(result_img)
        canvas.create_image(0, 0, anchor=NW, image=canvas_img)
        error_label.config(text='')
    except Exception as e:
        error_label.config(text=f"Error in region growing: {str(e)}")


def vis_jac():
    global result_binary, label_binary, canvas_img
    try:
        if result_binary is None:
            error_label.config(text="Please generate a segmentation first")
            return

        if label_binary is None:
            error_label.config(text=f"Label image '{label_img_file}' not found or couldn't be loaded")
            return

        # Create a base image for visualization
        if img is not None:
            # Use original grayscale image as base
            if len(np.array(img).shape) == 3:  # If RGB
                gray_img = np.array(img.convert('L'))
            else:
                gray_img = np.array(img)
        else:
            # Create blank grayscale image if original not available
            gray_img = np.zeros_like(np.array(result_img))

        # Create RGB visualization
        vis_img = np.stack([gray_img] * 3, axis=-1).astype(np.uint8)

        # Calculate different regions
        TP = np.logical_and(result_binary, label_binary)  # True Positives
        FP = np.logical_and(result_binary, ~label_binary)  # False Positives
        FN = np.logical_and(~result_binary, label_binary)  # False Negatives

        # Color code each region
        vis_img[TP] = [0, 255, 0]  # Green - correctly segmented
        vis_img[FP] = [255, 0, 0]  # Red - over-segmentation
        vis_img[FN] = [0, 0, 255]  # Blue - under-segmentation

        # Convert to PIL image and display
        error_vis_img = Image.fromarray(vis_img)
        canvas_img = ImageTk.PhotoImage(error_vis_img)
        canvas.create_image(0, 0, anchor=NW, image=canvas_img)

        # Save visualization if needed
        error_vis_img.save('jaccard_visualization.png')

        # Update error label with explanation
        error_label.config(text="Green: True Positive, Red: False Positive, Blue: False Negative", fg="black")

    except Exception as e:
        error_label.config(text=f"Error in Jaccard visualization: {str(e)}", fg="red")


def calculate_all_scores():
    global result_binary, label_binary
    try:
        if result_binary is None or label_binary is None:
            error_label.config(text="Need both prediction and ground truth for scoring")
            return

        # Calculate metrics using the pre-calculated binary masks
        jaccard = calculate_jaccard(result_binary, label_binary)
        dice = calculate_dice(result_binary, label_binary)
        sensitivity, specificity = calculate_sensitivity_specificity(result_binary, label_binary)
        precision, recall = calculate_precision_recall(result_binary, label_binary)

        # Create result message
        score_message = (f"Jaccard (IoU): {jaccard:.4f}\n"
                         f"Dice: {dice:.4f}\n"
                         f"Sensitivity: {sensitivity:.4f}\n"
                         f"Specificity: {specificity:.4f}\n"
                         f"Precision: {precision:.4f}\n"
                         f"Recall: {recall:.4f}")

        # Show in messagebox
        messagebox.showinfo("Segmentation Scores", score_message)

    except Exception as e:
        error_label.config(text=f"Error calculating scores: {str(e)}")


# Bind mouse click to get seed point
canvas.bind('<Button-1>', get_seed)

# Start the application
window.mainloop()
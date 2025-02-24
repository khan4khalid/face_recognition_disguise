import os
from tkinter import Tk, Canvas, Button, Label, Frame, Scrollbar, filedialog, VERTICAL, HORIZONTAL, RIGHT, BOTTOM, Y, X
from PIL import Image, ImageTk

# Configuration
PATCH_SIZE = (32,32)
OUTPUT_DIR = "classified_patches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_patches(image_path, patch_size):
    image = Image.open(image_path)
    width, height = image.size
    patches = []
    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):
            patch = image.crop((x, y, x + patch_size[0], y + patch_size[1]))
            patches.append(((x, y), patch))
    return image, patches

class PatchClassifierApp:
    def __init__(self, root):
        self.root = root
        self.image = None
        self.patches = []
        self.output_dir = OUTPUT_DIR
        self.selected_patches = set()
        self.patch_buttons = {}
        self.original_image_canvas = None

        # Main frames
        self.image_frame = Frame(root)
        self.image_frame.pack(side="left", padx=10)
        self.patches_frame = Frame(root)
        self.patches_frame.pack(side="right", padx=10, fill="both", expand=True)

        # Add scrollable canvas for the patches
        self.canvas = Canvas(self.patches_frame)
        self.v_scrollbar = Scrollbar(self.patches_frame, orient=VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = Scrollbar(self.patches_frame, orient=HORIZONTAL, command=self.canvas.xview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side=RIGHT, fill=Y)
        self.h_scrollbar.pack(side=BOTTOM, fill=X)

        # Select images button
        self.select_images_button = Button(root, text="Select Images", command=self.select_images)
        self.select_images_button.pack(pady=10)

        # Add finalize button
        self.finalize_button = Button(root, text="Finalize Selection", command=self.finalize_classification)
        self.finalize_button.pack(pady=10)

    def select_images(self):
        image_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if image_paths:
            self.image_file_paths = list(image_paths)
            self.process_next_image()

    def process_next_image(self):
        if not hasattr(self, 'image_file_paths') or not self.image_file_paths:
            print("No more images to process.")
            self.clear_original_image()
            return

        image_path = self.image_file_paths.pop(0)
        self.image, self.patches = generate_patches(image_path, PATCH_SIZE)
        self.display_original_image()
        self.display_patches_grid()

        # Extract person ID and image number from the file name
        base_name = os.path.basename(image_path)
        self.person_id, self.image_number = base_name.split("_")[0], base_name.split("_")[1]

    def display_original_image(self):
        self.clear_original_image()
        scaled_image = self.image.resize((self.image.width // 3, self.image.height // 3))
        img = ImageTk.PhotoImage(scaled_image)
        self.original_image_canvas = Canvas(self.image_frame, width=scaled_image.width, height=scaled_image.height)
        self.original_image_canvas.create_image(0, 0, anchor="nw", image=img)
        self.original_image_canvas.image = img
        self.original_image_canvas.pack()

    def clear_original_image(self):
        if self.original_image_canvas:
            self.original_image_canvas.destroy()
            self.original_image_canvas = None

    def display_patches_grid(self):
        """Display all patches in a scrollable grid."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for idx, (position, patch) in enumerate(self.patches):
            x, y = position
            patch_img = ImageTk.PhotoImage(patch.resize((PATCH_SIZE[0], PATCH_SIZE[1])))
            btn = Button(
                self.scrollable_frame,
                image=patch_img,
                command=lambda pos=position, idx=idx: self.toggle_patch_selection(pos, idx),
                relief="raised",
                bd=3,
            )
            btn.image = patch_img
            btn.grid(row=y // PATCH_SIZE[1], column=x // PATCH_SIZE[0], padx=2, pady=2)

            self.patch_buttons[position] = btn

            if position in self.selected_patches:
                btn.config(bg="#90EE90")

    def toggle_patch_selection(self, position, idx):
        """Toggle patch selection and update its visual state."""
        if position in self.selected_patches:
            self.selected_patches.remove(position)
            self.patch_buttons[position].config(bg="lightgray")
        else:
            self.selected_patches.add(position)
            self.patch_buttons[position].config(bg="#90EE90")

    def finalize_classification(self):
        """Save selected patches as clean and the rest as disguised."""
        for idx, (position, patch) in enumerate(self.patches):
            x, y = position
            patch_name = f"{self.person_id}_{self.image_number}_row{y // PATCH_SIZE[1]}_col{x // PATCH_SIZE[0]}.jpg"
            if position in self.selected_patches:
                classification = "clean"
            else:
                classification = "disguised"

            output_folder = os.path.join(self.output_dir, classification)
            os.makedirs(output_folder, exist_ok=True)
            patch.save(os.path.join(output_folder, patch_name))

        print(f"Patches for {self.person_id}_{self.image_number} saved.")

        self.selected_patches = set()
        self.process_next_image()

# Launch the application
root = Tk()
root.title("Patch Classifier")
root.geometry("1200x800")
app = PatchClassifierApp(root)
root.mainloop()

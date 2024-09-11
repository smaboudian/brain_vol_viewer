import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from ipywidgets import interact, fixed



# Function to load a NIfTI or MGZ file
def load_nifti(filepath):
    """
    Function to load a NIfTI or MGZ file using nibabel
    input: filepath to nifti, as str
    output: 3D numpy array, affine matrix
    """
    img = nib.load(filepath)
    return img.get_fdata(), img.affine

# Read the FreeSurfer LUT (Look-Up Table) and handle invalid lines
def load_lut(lut_file):
    """
    Function to load a FreeSurfer LUT file
    input: filepath to LUT file, as str
    output: dictionary with index as key and RGB tuple as value
    """
    lut = {}
    with open(lut_file) as f:
        lines = f.readlines()
        # Skip the header lines
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) == 6:
                index, label, r, g, b, a = parts
                lut[int(index)] = (int(r) / 255.0, int(g) / 255.0, int(b) / 255.0)
    return lut




# Function to display a slice with overlays, with the corrected orientation and color LUT applied
def show_slice_1view(t1_file, aparc_aseg_file, lut_file, custom_labels={},
               slice_num=0, axis='axial', alpha=1.0, display_aparc=True):
    """
    Function to display a slice with overlays, with the corrected orientation and color LUT applied.
    input: t1_file, aparc_aseg_file, lut_file, custom_labels (dict),
                slice_num (int), axis (str), alpha (float), display_aparc (bool)
    output: None

    """
    # Check if the axis is valid
    if axis not in ['axial', 'coronal', 'sagittal']:
        raise ValueError("Invalid axis specified. Use 'axial', 'coronal', or 'sagittal'.")

    # Load the T1 and aparc+aseg volumes
    t1_data, _ = load_nifti(t1_file)
    aparc_data, _ = load_nifti(aparc_aseg_file)

    # Load the LUT file
    lut = load_lut(lut_file)
    label_colors = {label: np.array(color) for label, color in lut.items()}

    # Select the slice
    if axis == 'coronal':  # Horizontal slice, viewed from above
        t1_slice = t1_data[:, :, slice_num].T        # .T: Transpose to correct orientation (90º)
        aparc_slice = aparc_data[:, :, slice_num].T  # .T: Transpose to correct orientation (90º)
        #t1_slice = np.flipud(t1_slice)               # optional: Flip vertically, ie by 180º
        #aparc_slice = np.flipud(aparc_slice)         # optional: Flip vertically, ie by 180º
    elif axis == 'axial':  # Vertical slice, viewed from front
        t1_slice = t1_data[:, slice_num, :].T        # .T: Transpose to correct orientation (90º)
        aparc_slice = aparc_data[:, slice_num, :].T  # .T: Transpose to correct orientation (90º)
        t1_slice = np.flipud(t1_slice)               # Flip vertically, ie by 180º
        aparc_slice = np.flipud(aparc_slice)         # Flip vertically, ie by 180º
    elif axis == 'sagittal':  # Vertical slice, viewed from the side
        t1_slice = t1_data[slice_num, :, :]
        aparc_slice = aparc_data[slice_num, :, :]

        
    #print(f"Slicing at: {axis.capitalize()} slice number {slice_num}")
    
    # Create an RGB overlay for aparc+aseg
    overlay = np.zeros(t1_slice.shape + (3,), dtype=np.float32)

    if display_aparc:
        for label_value, color in label_colors.items():
            overlay[aparc_slice == label_value] = color

    # Create a mask overlay for custom labels
    mask_overlay = np.zeros(t1_slice.shape + (3,), dtype=np.float32)

    for custom_label_file, custom_color in custom_labels.items():        
        if custom_label_file:
            custom_label_data, _ = load_nifti(custom_label_file)
            
            # Select the appropriate slice based on the axis
            if axis == 'coronal':
                custom_label_slice = custom_label_data[:, :, slice_num].T  # .T: Transpose to correct orientation (90º)
                #custom_label_slice = np.flipud(custom_label_slice)         # Flip vertically, ie by 180º
            elif axis == 'axial':
                custom_label_slice = custom_label_data[:, slice_num, :].T  # .T: Transpose to correct orientation (90º)
                custom_label_slice = np.flipud(custom_label_slice)         # Flip vertically, ie by 180º
            elif axis == 'sagittal':
                custom_label_slice = custom_label_data[slice_num, :, :]
                
            # Ensure the custom label values are binary
            custom_label_slice = np.where(custom_label_slice == 1, 1, 0)
            
            # Apply custom masks
            mask_overlay[custom_label_slice == 1] = custom_color

    # Normalize the mask overlay for display
    mask_overlay_max = np.max(mask_overlay)
    if mask_overlay_max > 0:
        mask_overlay /= mask_overlay_max

    # Combine T1 slice with overlays
    combined_overlay = alpha * overlay + (1 - alpha) * mask_overlay

    # Show the slice
    plt.figure(figsize=(10, 10))
    plt.imshow(np.flipud(t1_slice), cmap='gray', origin='lower')
    plt.imshow(np.flipud(combined_overlay), alpha=alpha, origin='lower')
    plt.colorbar()
    plt.title(f"Slice {slice_num} - {axis.capitalize()}")
    plt.show()

    

# Create an interactive widget to scroll through the slices of one view of a T1 volume (axial, coronal, or sagittal)
def scroll_slices_1view(t1_file, aparc_aseg_file, lut_file, custom_labels={},
                  axis='axial', alpha=0.5, display_aparc=True):
    """
    Function to create an interactive widget to scroll through the slices of one view of a T1 volume (axial, coronal, or sagittal).
    Calls show_slice_1view function.
    Input: t1_file, aparc_aseg_file, lut_file, custom_labels (dict), axis (str), alpha (float), display_aparc (bool)
    Output: None
    """
    axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
    if axis not in axis_map:
        raise ValueError("Invalid axis specified. Use 'axial', 'coronal', or 'sagittal'.")
    
    # Load the T1 data to determine the middle slice
    t1_data, _ = load_nifti(t1_file)
    axis_size = t1_data.shape[axis_map[axis]]
    middle_slice = axis_size // 2  # Calculate the middle slice
    
    # Create an interactive widget to scroll through the slices
    interact(show_slice_1view, 
             t1_file=fixed(t1_file), aparc_aseg_file=fixed(aparc_aseg_file), 
             lut_file=fixed(lut_file), custom_labels=fixed(custom_labels),
             #slice_num=(0, axis_size - 1, 1), 
             slice_num=middle_slice, # Set initial slice to middle
             axis=fixed(axis), alpha=fixed(alpha),
             display_aparc=fixed(display_aparc),)  







# Function to display slices from all 3 views simultaneously (coronal, axial, sagittal)
def show_slices_3views(t1_file, aparc_aseg_file, lut_file, custom_labels={},
                slice_num_axial=0, slice_num_coronal=0, slice_num_sagittal=0,
                alpha=1.0, display_aparc=True):
    """
    Function to display slices from all 3 views simultaneously (coronal, axial, sagittal).
    input: t1_file, aparc_aseg_file, lut_file, custom_labels (dict), 
              slice_num_axial, slice_num_coronal, slice_num_sagittal (int),
              alpha (float), display_aparc (bool)   
    output: None
    """
    # Load the T1 and aparc+aseg volumes
    t1_data, _ = load_nifti(t1_file)
    aparc_data, _ = load_nifti(aparc_aseg_file)

    # Load the LUT file
    lut = load_lut(lut_file)
    label_colors = {label: np.array(color) for label, color in lut.items()}

    # Function to get a slice based on axis and slice number
    def get_slice(data, slice_num, axis):
        if axis == 'coronal':
            return data[:, :, slice_num].T
        elif axis == 'axial':
            return np.flipud(data[:, slice_num, :].T)
        elif axis == 'sagittal':
            return data[slice_num, :, :]
        else:
            raise ValueError("Invalid axis specified.")
 
    # Function to create overlay from aparc+aseg data
    def create_overlay(slice_data, aparc_slice):
        overlay = np.zeros(slice_data.shape + (3,), dtype=np.float32)
        if display_aparc:
            for label_value, color in label_colors.items():
                overlay[aparc_slice == label_value] = color
        return overlay

    # Function to apply custom labels
    def apply_custom_labels(slice_data, custom_labels_dict, axis):
        mask_overlay = np.zeros(slice_data.shape + (3,), dtype=np.float32)
        for custom_label_file, custom_color in custom_labels_dict.items():
            if custom_label_file:
                custom_label_data, _ = load_nifti(custom_label_file)
                # Slice the custom label data according to the axis
                if axis == 'axial':
                    custom_label_slice = get_slice(custom_label_data, slice_num_axial, 'axial')
                elif axis == 'coronal':
                    custom_label_slice = get_slice(custom_label_data, slice_num_coronal, 'coronal')
                elif axis == 'sagittal':
                    custom_label_slice = get_slice(custom_label_data, slice_num_sagittal, 'sagittal')
                else:
                    raise ValueError("Invalid axis specified.")
                custom_label_slice = np.where(custom_label_slice == 1, 1, 0)
                mask_overlay[custom_label_slice == 1] = custom_color
        mask_overlay_max = np.max(mask_overlay)
        if mask_overlay_max > 0:
            mask_overlay /= mask_overlay_max
        return mask_overlay

    # Axial slice
    t1_slice_axial = get_slice(t1_data, slice_num_axial, 'axial')
    aparc_slice_axial = get_slice(aparc_data, slice_num_axial, 'axial')
    overlay_axial = create_overlay(t1_slice_axial, aparc_slice_axial)
    mask_overlay_axial = apply_custom_labels(t1_slice_axial, custom_labels, 'axial')
    combined_overlay_axial = alpha * overlay_axial + (1 - alpha) * mask_overlay_axial

    # Coronal slice
    t1_slice_coronal = get_slice(t1_data, slice_num_coronal, 'coronal')
    aparc_slice_coronal = get_slice(aparc_data, slice_num_coronal, 'coronal')
    overlay_coronal = create_overlay(t1_slice_coronal, aparc_slice_coronal)
    mask_overlay_coronal = apply_custom_labels(t1_slice_coronal, custom_labels, 'coronal')
    combined_overlay_coronal = alpha * overlay_coronal + (1 - alpha) * mask_overlay_coronal

    # Sagittal slice
    t1_slice_sagittal = get_slice(t1_data, slice_num_sagittal, 'sagittal')
    aparc_slice_sagittal = get_slice(aparc_data, slice_num_sagittal, 'sagittal')
    overlay_sagittal = create_overlay(t1_slice_sagittal, aparc_slice_sagittal)
    mask_overlay_sagittal = apply_custom_labels(t1_slice_sagittal, custom_labels, 'sagittal')
    combined_overlay_sagittal = alpha * overlay_sagittal + (1 - alpha) * mask_overlay_sagittal

    # Plotting all three views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.flipud(t1_slice_axial), cmap='gray', origin='lower')
    axes[0].imshow(np.flipud(combined_overlay_axial), alpha=alpha, origin='lower')
    axes[0].set_title(f"Axial Slice {slice_num_axial}")

    axes[1].imshow(np.flipud(t1_slice_coronal), cmap='gray', origin='lower')
    axes[1].imshow(np.flipud(combined_overlay_coronal), alpha=alpha, origin='lower')
    axes[1].set_title(f"Coronal Slice {slice_num_coronal}")

    axes[2].imshow(np.flipud(t1_slice_sagittal), cmap='gray', origin='lower')
    axes[2].imshow(np.flipud(combined_overlay_sagittal), alpha=alpha, origin='lower')
    axes[2].set_title(f"Sagittal Slice {slice_num_sagittal}")

    plt.tight_layout()
    plt.show()




# Create an interactive widget to scroll through the slices for all 3 views (coronal, axial, sagittal)
def scroll_slices_3views(t1_file, aparc_aseg_file, lut_file, custom_labels={},
                  alpha=0.5, display_aparc=True):
    """
    Function to create an interactive widget to scroll through the slices for all 3 views (coronal, axial, sagittal).
    Calls show_slices_3views function.
    input: t1_file, aparc_aseg_file, lut_file, custom_labels (dict),
              alpha (float), display_aparc (bool)           
    output: None    
    """
    # Load the T1 data to determine slice range
    t1_data, _ = load_nifti(t1_file)
    axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
    
    # Define the range for scrolling
    slice_ranges = {axis: range(t1_data.shape[axis_map[axis]]) for axis in axis_map.keys()}
    
    interact(show_slices_3views,
             t1_file=fixed(t1_file), aparc_aseg_file=fixed(aparc_aseg_file),
             lut_file=fixed(lut_file), custom_labels=fixed(custom_labels),
             slice_num_axial=(0, len(slice_ranges['axial']) - 1, 1),
             slice_num_coronal=(0, len(slice_ranges['coronal']) - 1, 1),
             slice_num_sagittal=(0, len(slice_ranges['sagittal']) - 1, 1),
             alpha=fixed(alpha),
             display_aparc=fixed(display_aparc))


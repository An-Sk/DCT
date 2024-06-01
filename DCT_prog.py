import cv2
import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio
import sys
import os


def main():
    """
  Main function to perform text encoding in an image using DCT.
  """
    # Get user inputs (text and image path) from command line arguments
    text_message = sys.argv[1]
    image_path = sys.argv[2]

    # Read and resize the image
    origin_image = cv2.imread(image_path)
    resized_image = resize_image(origin_image)
    # Convert image to grayscale (or use the luminance channel if working with YCbCr)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # print(gray_image.shape)
    # Perform DCT on the image, assuming having 3 color channels
    dct_coefficients = perform_dct(gray_image)
    #print(dct_coefficients.shape, dct_coefficients)
    # Encode text message in DCT coefficients
    modified_coefficients = encode_text_in_dct(dct_coefficients.copy(), text_message)

    # Perform inverse DCT to reconstruct the modified image
    modified_image = perform_idct(modified_coefficients)

    # Decode the encoded message
    decoded_message = decode_text_from_dct(modified_coefficients)
    print(f"Decoded message: {decoded_message}")
    # Calculate PSNR between original and modified images
    psnr = calculate_psnr(gray_image, modified_image)

    # Display both images side-by-side
    cv2.imwrite("gray_image.bmp", gray_image)
    cv2.imwrite("modified_image.bmp", modified_image)

    print(f"PSNR between original and modified image: {psnr:.2f} dB")
    display_images(gray_image, modified_image)
    # Print the PSNR value



def resize_image(image, target_size=(32, 32)):
    """
  Resizes the image to a suitable size for DCT (divisible by 8).
  """
    h, w = image.shape[:2]
    new_h, new_w = (target_size[0] - h % target_size[0], target_size[1] - w % target_size[1])
    resized_image = cv2.resize(image, (w + new_w, h + new_h))

    # Get the script directory (where the program is running)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #print(script_dir)
    # Combine script directory and output filename
    #print(os.path.isdir(script_dir), end='\n')
    #output_path = os.path.join(script_dir, output_filename)
    #print(output_path)
    #print(fr"{output_path}")
    #print(r"F:\hse\ПА средства защиты информации\лабы\lr_5\resized_image.jpg")
    # Save the resized image, no fucking clue why path does not work

    return resized_image


def perform_dct(image):
    """
    Performs the Discrete Cosine Transform (DCT) on the image.
    """

    # Get image dimensions
    height, width = image.shape

    # Ensure the image dimensions are multiples of 8
    height -= height % 8
    width -= width % 8
    gray_image = image[:height, :width]

    # Break image into 8x8 non-overlapping blocks
    blocks = np.array([gray_image[i:i + 8, j:j + 8] for i in range(0, height, 8) for j in range(0, width, 8)])
    blocks = blocks.reshape(-1, 8, 8)  # Reshape to have each 8x8 block in the right format

    # Apply 2D DCT on each block
    dct_coefficients = np.zeros_like(blocks, dtype=float)
    for k in range(blocks.shape[0]):
        dct_coefficients[k] = dctn(blocks[k], type=2)

    # Reshape DCT coefficients back to the image shape
    dct_image = np.vstack([np.hstack(dct_coefficients[i * (width // 8):(i + 1) * (width // 8)])
                           for i in range(height // 8)])

    return dct_image


def perform_idct(dct_coefficients):
    """
    Performs the Inverse Discrete Cosine Transform (IDCT) on the coefficients.
    """
    # Get the block size (assuming 8x8 blocks)
    block_size = 8

    # Calculate the number of blocks
    num_blocks_vertical = dct_coefficients.shape[0] // block_size
    num_blocks_horizontal = dct_coefficients.shape[1] // block_size

    # Create an empty array to store the reconstructed image
    reconstructed_image = np.zeros((dct_coefficients.shape[0], dct_coefficients.shape[1]))

    # Apply inverse DCT on each block
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = dct_coefficients[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            reconstructed_image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = idctn(block,
                                                                                                                  type=2)

    return reconstructed_image


def encode_text_in_dct(coefficients, text_message):
    """
    Embeds the text message into the DCT coefficients.

    This is a simplified example. A more robust strategy is needed for real use.
    """
    # Convert text to binary string
    binary_message = "".join(format(ord(char), '08b') for char in text_message)
    binary_message += '1111111111111110'  # End of text marker

    # Flatten the coefficients array to easily modify its elements
    flat_coefficients = coefficients.flatten()

    # Ensure the flat_coefficients array is of integer type for bitwise operations
    int_coefficients = flat_coefficients.astype(np.int32)

    # Select modification locations (e.g., starting from the end to avoid low-frequency coefficients)
    modification_locations = np.argsort(np.abs(int_coefficients))[::-1]

    # Modify coefficient LSB based on message bits
    message_index = 0
    for loc in modification_locations:
        if message_index < len(binary_message):
            int_coefficients[loc] = (int_coefficients[loc] & ~1) | int(binary_message[message_index])
            message_index += 1
        else:
            break

    # Convert coefficients back to their original type
    modified_coefficients = int_coefficients.astype(coefficients.dtype)
    modified_coefficients = modified_coefficients.reshape(coefficients.shape)
    return modified_coefficients

def decode_text_from_dct(coefficients):
    """
    Extracts the text message hidden in the DCT coefficients.
    """
    flat_coefficients = coefficients.flatten()
    int_coefficients = flat_coefficients.astype(np.int32)

    modification_locations = np.argsort(np.abs(int_coefficients))[::-1]

    binary_message = []
    for loc in modification_locations:
        lsb = int_coefficients[loc] & 1
        binary_message.append(str(lsb))

        if len(binary_message) >= 16 and ''.join(binary_message[-16:]) == '1111111111111110':
            binary_message = binary_message[:-16]
            break

    binary_message = ''.join(binary_message)
    text_message = ''.join(chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8))
    return text_message

def calculate_psnr(image1, image2):
    """
  Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
  """
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite for identical images
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    return psnr


def display_images(image1, image2):
    """
  Displays two images side-by-side using matplotlib.
  """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image1), plt.title("Original Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image2), plt.title("Modified Image")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

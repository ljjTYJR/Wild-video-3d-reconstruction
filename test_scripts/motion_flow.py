# test the function of computing the motion flow between two images
from dpvo.utils import measure_motion
import cv2
import numpy as np
import os
import glob
import json

def collect_images_with_sufficient_flow(image_dir, flow_threshold=2.0, target_count=50, output_dir='flow_results', save_flow_vis=False):
    """
    Continuously detect optical flow between images in a directory.
    Collect image pairs with sufficient flow.
    
    Args:
        image_dir: Directory containing images
        flow_threshold: Minimum flow magnitude to consider sufficient
        target_count: Number of image pairs to collect
        output_dir: Directory to save results
        save_flow_vis: Whether to save flow visualizations (default False)
    """
    
    # Get all image files sorted by name
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files.sort()
    
    if len(image_files) < 2:
        print(f"Not enough images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    collected_pairs = []
    flow_data = []
    
    i = 0  # Current base image index
    while i < len(image_files) and len(collected_pairs) < target_count:
        base_img_path = image_files[i]
        base_img = cv2.imread(base_img_path)
        
        if base_img is None:
            print(f"Failed to load {base_img_path}")
            i += 1
            continue
            
        print(f"Processing base image {i+1}/{len(image_files)}: {os.path.basename(base_img_path)}")
        
        # Try subsequent images until we find sufficient flow
        found_match = False
        for offset in range(1, min(20, len(image_files) - i)):  # Limit search range
            compare_idx = i + offset
            compare_img_path = image_files[compare_idx]
            compare_img = cv2.imread(compare_img_path)
            
            if compare_img is None:
                continue
            
            # Measure motion between images
            mean_flow, median_flow, mean_norm, median_norm, motion_ratio = measure_motion(
                base_img, compare_img, threshold=1.0
            )
            
            print(f"  vs image {compare_idx+1} (offset {offset}): mean_flow={mean_flow:.2f}")
            
            if mean_flow >= flow_threshold:
                # Found sufficient flow
                pair_info = {
                    'pair_id': len(collected_pairs),
                    'base_image': base_img_path,
                    'compare_image': compare_img_path,
                    'image_offset': offset,
                    'mean_flow': float(mean_flow),
                    'median_flow': float(median_flow),
                    'mean_norm': float(mean_norm),
                    'median_norm': float(median_norm),
                    'motion_ratio': float(motion_ratio)
                }
                
                collected_pairs.append(pair_info)
                flow_data.append(pair_info)
                
                print(f"  ✓ Collected pair {len(collected_pairs)}/{target_count}")
                print(f"    Flow: {mean_flow:.3f}, Motion ratio: {motion_ratio:.3f}")
                
                # Save original images with their original names
                base_name = os.path.basename(base_img_path)
                compare_name = os.path.basename(compare_img_path)
                
                cv2.imwrite(os.path.join(output_dir, base_name), base_img)
                cv2.imwrite(os.path.join(output_dir, compare_name), compare_img)
                
                # Optionally save flow visualization
                if save_flow_vis:
                    save_flow_visualization(base_img, compare_img, len(collected_pairs)-1, output_dir)
                
                # Move to the compare image as next base
                i = compare_idx
                found_match = True
                break
        
        if not found_match:
            print(f"  No sufficient flow found for {os.path.basename(base_img_path)}")
            i += 1
    
    # Save collected data
    with open(os.path.join(output_dir, 'flow_data.json'), 'w') as f:
        json.dump(flow_data, f, indent=2)
    
    print(f"\nCollection completed: {len(collected_pairs)}/{target_count} pairs collected")
    print(f"Results saved to {output_dir}/")
    
    # Print summary statistics
    if collected_pairs:
        flows = [p['mean_flow'] for p in collected_pairs]
        print(f"Flow statistics:")
        print(f"  Mean: {np.mean(flows):.3f}")
        print(f"  Std: {np.std(flows):.3f}")
        print(f"  Min: {np.min(flows):.3f}")
        print(f"  Max: {np.max(flows):.3f}")
    
    return collected_pairs

def save_flow_visualization(img1, img2, pair_id, output_dir):
    """Save flow visualization for an image pair"""
    try:
        # Convert to grayscale for flow computation
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Compute flow for visualization
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Create flow visualization
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        angle = np.arctan2(flow[:,:,1], flow[:,:,0])
        
        # Create HSV representation
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[:,:,0] = (angle + np.pi) * 180 / (2 * np.pi)  # Hue
        hsv[:,:,1] = 255  # Saturation
        hsv[:,:,2] = np.clip(magnitude * 10, 0, 255)  # Value
        
        # Convert to BGR for display
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Create a combined visualization showing both images and flow
        h, w = img1.shape[:2]
        combined = np.zeros((h, w * 3, 3), dtype=np.uint8)

        # Place first image
        combined[:, :w] = img1

        # Place second image
        combined[:, w:2*w] = img2

        # Place flow visualization
        combined[:, 2*w:3*w] = flow_vis

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f'Image A', (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f'Image B', (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f'Flow {pair_id:02d}', (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)

        # Save combined visualization
        output_path = os.path.join(output_dir, f'flow_pair_{pair_id:03d}.jpg')
        cv2.imwrite(output_path, combined)

    except Exception as e:
        print(f"Could not save flow visualization for pair {pair_id}: {e}")

# Main execution
if __name__ == "__main__":
    # Configuration
    image_directory = '/media/shuo/T7/rgbd_slam/euroc/MH_03_medium/mav0/cam0/data/'
    flow_threshold = 2.0  # Minimum mean flow magnitude
    target_pairs = 50     # Number of pairs to collect
    
    print(f"Starting optical flow collection:")
    print(f"  Image directory: {image_directory}")
    print(f"  Flow threshold: {flow_threshold}")
    print(f"  Target pairs: {target_pairs}")
    print()
    
    # Collect images with sufficient flow
    collected = collect_images_with_sufficient_flow(
        image_dir=image_directory,
        flow_threshold=flow_threshold,
        target_count=target_pairs,
        output_dir='flow_results',
        save_flow_vis=False  # Set to True if you want flow visualizations
    )
    
    print("\nOptical flow collection completed.")
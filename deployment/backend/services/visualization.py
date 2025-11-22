"""Visualization generation service"""
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def generate_comprehensive_visualizations(
    detections: List[Dict],
    class_metrics: Dict,
    confidence_metrics: Dict,
    spatial_metrics: Dict,
    img_width: int,
    img_height: int
) -> Dict[str, str]:
    """
    Generate all visualizations as base64 encoded images
    
    Args:
        detections: List of detection dictionaries
        class_metrics: Class-specific metrics
        confidence_metrics: Confidence analysis
        spatial_metrics: Spatial distribution metrics
        img_width: Image width
        img_height: Image height
        
    Returns:
        Dictionary of visualization name -> base64 encoded image
    """
    visualizations = {}
    
    if not detections:
        return visualizations
    
    # 1. Class Distribution Bar Chart
    visualizations['class_distribution'] = generate_class_distribution(class_metrics)
    
    # 2. Confidence Distribution Histogram
    visualizations['confidence_distribution'] = generate_confidence_histogram(detections)
    
    # 3. Spatial Distribution Heatmap
    visualizations['spatial_heatmap'] = generate_spatial_heatmap(
        detections, img_width, img_height
    )
    
    # 4. Box Plot: Confidence by Class
    visualizations['confidence_by_class'] = generate_confidence_by_class(
        detections, class_metrics
    )
    
    # 5. Area Distribution Scatter
    visualizations['area_vs_confidence'] = generate_area_vs_confidence(detections)
    
    # 6. Quadrant Distribution Pie Chart
    visualizations['quadrant_distribution'] = generate_quadrant_pie(spatial_metrics)
    
    # 7. Size Distribution
    visualizations['size_distribution'] = generate_size_distribution(spatial_metrics)
    
    # 8. Top Classes by Average Confidence
    visualizations['top_classes_confidence'] = generate_top_classes(class_metrics)
    
    return visualizations


def generate_class_distribution(class_metrics: Dict) -> str:
    """Generate class distribution bar chart"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        classes = list(class_metrics.keys())
        counts = [class_metrics[c]['count'] for c in classes]
        
        bars = ax.bar(range(len(classes)), counts, color='steelblue', alpha=0.8)
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        result = fig_to_base64(fig)
        plt.close(fig)
        return result
    except Exception as e:
        print(f"Error generating class distribution: {e}")
        return ""


def generate_confidence_histogram(detections: List[Dict]) -> str:
    """Generate confidence distribution histogram"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        confidences = [d['confidence'] for d in detections]
        
        ax.hist(confidences, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        ax.axvline(np.median(confidences), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(confidences):.3f}')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        result = fig_to_base64(fig)
        plt.close(fig)
        return result
    except Exception as e:
        print(f"Error generating confidence distribution: {e}")
        return ""


def generate_spatial_heatmap(
    detections: List[Dict],
    img_width: int,
    img_height: int
) -> str:
    """Generate spatial distribution heatmap"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        centers_x = [d['bbox']['center_x'] for d in detections]
        centers_y = [d['bbox']['center_y'] for d in detections]
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(centers_x, centers_y, bins=20)
        
        im = ax.imshow(heatmap.T, origin='lower', cmap='YlOrRd', aspect='auto',
                      extent=[0, img_width, 0, img_height])
        
        ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax.set_title('Spatial Distribution Heatmap', fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Detection Density')
        plt.tight_layout()
        result = fig_to_base64(fig)
        plt.close(fig)
        return result
    except Exception as e:
        print(f"Error generating spatial heatmap: {e}")
        return ""


def generate_confidence_by_class(
    detections: List[Dict],
    class_metrics: Dict
) -> str:
    """Generate confidence distribution by class boxplot"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        class_conf_data = []
        class_labels = []
        
        # Get top 10 classes by count
        sorted_classes = sorted(
            class_metrics.keys(),
            key=lambda x: class_metrics[x]['count'],
            reverse=True
        )[:10]
        
        for cls in sorted_classes:
            conf_values = [d['confidence'] for d in detections if d['class_name'] == cls]
            class_conf_data.append(conf_values)
            class_labels.append(f"{cls}\n(n={len(conf_values)})")
        
        # Create boxplot
        bp = ax.boxplot(class_conf_data, labels=class_labels, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Distribution by Class (Top 10)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        result = fig_to_base64(fig)
        plt.close(fig)
        return result
    except Exception as e:
        print(f"Error generating confidence by class: {e}")
        return ""


def generate_area_vs_confidence(detections: List[Dict]) -> str:
    """Generate area vs confidence scatter plot"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        areas = [d['area'] for d in detections]
        confidences = [d['confidence'] for d in detections]
        
        scatter = ax.scatter(areas, confidences, alpha=0.6, c=confidences, 
                           cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Detection Area (pixels²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Detection Area vs Confidence', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Confidence')
        plt.tight_layout()
        result = fig_to_base64(fig)
        plt.close(fig)
        return result
    except Exception as e:
        print(f"Error generating area vs confidence: {e}")
        return ""


def generate_quadrant_pie(spatial_metrics: Dict) -> str:
    """Generate quadrant distribution pie chart"""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        quadrants = spatial_metrics.get('quadrant_distribution', {})
        if not quadrants:
            plt.close(fig)
            return ""
            
        labels = [f'{k}\n({v} elements)' for k, v in quadrants.items()]
        sizes = list(quadrants.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Detection Distribution by Quadrant', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        result = fig_to_base64(fig)
        plt.close(fig)
        return result
    except Exception as e:
        print(f"Error generating quadrant distribution: {e}")
        return ""


def generate_size_distribution(spatial_metrics: Dict) -> str:
    """Generate size distribution bar chart"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        size_dist = spatial_metrics.get('size_distribution', {})
        if not size_dist:
            plt.close(fig)
            return ""
            
        categories = list(size_dist.keys())
        values = list(size_dist.values())
        
        bars = ax.bar(categories, values, 
                     color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], 
                     alpha=0.8)
        ax.set_xlabel('Size Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Detection Size Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        result = fig_to_base64(fig)
        plt.close(fig)
        return result
    except Exception as e:
        print(f"Error generating size distribution: {e}")
        return ""


def generate_top_classes(class_metrics: Dict) -> str:
    """Generate top classes by confidence horizontal bar chart"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_classes = sorted(
            class_metrics.items(),
            key=lambda x: x[1]['confidence_stats']['mean'],
            reverse=True
        )[:15]
        
        classes = [c[0] for c in sorted_classes]
        avg_confs = [c[1]['confidence_stats']['mean'] for c in sorted_classes]
        
        bars = ax.barh(range(len(classes)), avg_confs, color='coral', alpha=0.8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_xlabel('Average Confidence', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Classes by Average Confidence', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{avg_confs[i]:.3f}', ha='left', va='center', 
                   fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        result = fig_to_base64(fig)
        plt.close(fig)
        return result
    except Exception as e:
        print(f"Error generating top classes: {e}")
        return ""


def fig_to_base64(fig) -> str:
    """
    Convert matplotlib figure to base64 encoded string
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return f"data:image/png;base64,{image_base64}"
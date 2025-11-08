import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star" | "line";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    const { width, height, data } = imageData;

    // Step 1: Convert to binary image (threshold)
    const binary = this.toBinaryImage(data, width, height);

    // Step 2: Find connected components (each component = potential shape)
    const components = this.findConnectedComponents(binary, width, height);

    // Step 3: Process each component
    const shapes: DetectedShape[] = [];

    for (const pixels of components) {
      // Filter out noise (very small components)
      if (pixels.length < 20) continue;

      // Extract features
      const features = this.computeFeatures(pixels, width, height);

      // Filter out shapes that are too large (likely the image border)
      const imageArea = width * height;
      if (features.area > imageArea * 0.8) continue;

      // Extract contour (boundary pixels)
      const contour = this.extractContour(pixels, width, height);

      // Order contour points (trace the boundary in order)
      const orderedContour = this.orderContourPoints(contour);

      // Detect corners using a more robust method
      const corners = this.detectCorners(orderedContour, features.area);

      // Simplify the contour for general geometry analysis, but use corners for vertex count
      const simplifiedPolygon = this.simplifyContour(orderedContour, 2.0);

      // Calculate additional geometric features
      const geometricFeatures = this.analyzeGeometry(
        simplifiedPolygon,
        features.area,
        features.perimeter,
        corners.length // Use the count from our robust corner detector
      );

      // Classify the shape
      const classification = this.classifyShape(geometricFeatures);

      // Debug logging
      console.log("Shape Debug:", {
        center: features.center,
        area: features.area,
        raw_contour_points: contour.length,
        ordered_contour_points: orderedContour.length,
        corners_detected: corners.length,
        simplified_vertices: simplifiedPolygon.length,
        final_vertices: geometricFeatures.vertices,
        circularity: geometricFeatures.circularity,
        aspectRatio: geometricFeatures.aspectRatio,
        solidity: geometricFeatures.solidity,
        isConvex: geometricFeatures.isConvex,
        classification: classification.type,
        confidence: classification.confidence,
      });

      if (classification.type) {
        shapes.push({
          type: classification.type,
          confidence: classification.confidence,
          boundingBox: {
            x: features.minX,
            y: features.minY,
            width: features.width,
            height: features.height,
          },
          center: features.center,
          area: features.area,
        });
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: width,
      imageHeight: height,
    };
  }

  /**
   * Convert RGBA image to binary (black & white)
   * Pixels below threshold = shape (1), above = background (0)
   */
  private toBinaryImage(
    data: Uint8ClampedArray,
    width: number,
    height: number
  ): Uint8Array {
    const binary = new Uint8Array(width * height);
    const threshold = 128;

    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];

      // Convert to grayscale
      const gray = (r + g + b) / 3;

      // Threshold: dark pixels are shapes
      binary[i] = gray < threshold ? 1 : 0;
    }

    return binary;
  }

  /**
   * Find connected components using flood fill (BFS)
   * Each component is a potential shape
   */
  private findConnectedComponents(
    binary: Uint8Array,
    width: number,
    height: number
  ): Point[][] {
    const visited = new Set<number>();
    const components: Point[][] = [];

    // Flood fill from a starting point
    const floodFill = (startX: number, startY: number): Point[] => {
      const queue: Point[] = [{ x: startX, y: startY }];
      const pixels: Point[] = [];

      while (queue.length > 0) {
        const point = queue.shift()!;
        const { x, y } = point;

        // Boundary checks
        if (x < 0 || y < 0 || x >= width || y >= height) continue;

        const idx = y * width + x;

        // Skip if already visited or background
        if (visited.has(idx) || binary[idx] === 0) continue;

        visited.add(idx);
        pixels.push({ x, y });

        // Add 4-connected neighbors
        queue.push({ x: x + 1, y });
        queue.push({ x: x - 1, y });
        queue.push({ x, y: y + 1 });
        queue.push({ x, y: y - 1 });
      }

      return pixels;
    };

    // Scan entire image for unvisited shape pixels
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (binary[idx] === 1 && !visited.has(idx)) {
          const component = floodFill(x, y);
          if (component.length > 0) {
            components.push(component);
          }
        }
      }
    }

    return components;
  }

  /**
   * Compute geometric features for a component
   */
  private computeFeatures(
    pixels: Point[],
    width: number,
    height: number
  ): {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
    width: number;
    height: number;
    area: number;
    center: Point;
    perimeter: number;
  } {
    const xs = pixels.map((p) => p.x);
    const ys = pixels.map((p) => p.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const w = maxX - minX + 1;
    const h = maxY - minY + 1;

    const centerX = xs.reduce((a, b) => a + b, 0) / xs.length;
    const centerY = ys.reduce((a, b) => a + b, 0) / ys.length;

    // Estimate perimeter (boundary pixels)
    const perimeter = this.estimatePerimeter(pixels);

    return {
      minX,
      minY,
      maxX,
      maxY,
      width: w,
      height: h,
      area: pixels.length,
      center: { x: centerX, y: centerY },
      perimeter,
    };
  }

  /**
   * Estimate perimeter by counting boundary pixels
   */
  private estimatePerimeter(pixels: Point[]): number {
    const pixelSet = new Set(pixels.map((p) => `${p.x},${p.y}`));
    let boundaryCount = 0;

    for (const pixel of pixels) {
      const { x, y } = pixel;

      // Check if any neighbor is not in the shape
      const neighbors = [
        `${x + 1},${y}`,
        `${x - 1},${y}`,
        `${x},${y + 1}`,
        `${x},${y - 1}`,
      ];

      if (neighbors.some((n) => !pixelSet.has(n))) {
        boundaryCount++;
      }
    }

    return boundaryCount;
  }

  /**
   * Extract contour (boundary) pixels from component
   */
  private extractContour(
    pixels: Point[],
    width: number,
    height: number
  ): Point[] {
    const contour: Point[] = [];
    const pixelSet = new Set(pixels.map((p) => `${p.x},${p.y}`));

    for (const pixel of pixels) {
      let boundaryCount = 0;
      // Check 8 neighbors
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = pixel.x + dx;
          const ny = pixel.y + dy;
          if (!pixelSet.has(`${nx},${ny}`)) {
            boundaryCount++;
          }
        }
      }
      if (boundaryCount > 0) {
        contour.push(pixel);
      }
    }

    return contour;
  }

  /**
   * Orders contour points using Moore-Neighbor tracing algorithm.
   * This is essential for correctly tracing complex, non-convex shapes.
   */
  private orderContourPoints(contour: Point[]): Point[] {
    if (contour.length < 2) return contour;

    const contourSet = new Set(contour.map((p) => `${p.x},${p.y}`));
    let startPoint = contour.reduce((p1, p2) => (p1.y < p2.y ? p1 : p2.y === p2.y && p1.x < p2.x ? p1 : p2));

    const ordered: Point[] = [];
    let current = startPoint;
    let backTrack = { x: current.x - 1, y: current.y }; // A point guaranteed not to be on the contour to start

    const neighbors = [
      { dx: 0, dy: -1 }, { dx: 1, dy: -1 }, { dx: 1, dy: 0 }, { dx: 1, dy: 1 },
      { dx: 0, dy: 1 }, { dx: -1, dy: 1 }, { dx: -1, dy: 0 }, { dx: -1, dy: -1 },
    ];

    do {
      ordered.push(current);
      let backTrackIndex = neighbors.findIndex(n => (current.x + n.dx === backTrack.x && current.y + n.dy === backTrack.y));
      if (backTrackIndex === -1) backTrackIndex = 6; // Start search from left if backtrack not found

      let next: Point | null = null;
      for (let i = 1; i <= 8; i++) {
        const neighborOffset = neighbors[(backTrackIndex + i) % 8];
        const nextCandidate = { x: current.x + neighborOffset.dx, y: current.y + neighborOffset.dy };
        if (contourSet.has(`${nextCandidate.x},${nextCandidate.y}`)) {
          next = nextCandidate;
          backTrack = { x: current.x, y: current.y };
          break;
        }
      }

      if (next) {
        current = next;
      } else {
        // Isolated pixel or end of line
        break;
      }
    } while (current.x !== startPoint.x || current.y !== startPoint.y);

    return ordered;
  }

  /**
   * Detects corners in an ordered contour by analyzing curvature.
   */
  private detectCorners(contour: Point[], area: number): Point[] {
    if (contour.length < 10) return contour;

    const corners: Point[] = [];
    // Use even smaller window for better corner detection
    const windowSize = Math.max(2, Math.min(10, Math.floor(Math.sqrt(area) * 0.1)));
    const angleThreshold = Math.PI * 0.7; // ~126 degrees - very permissive

    for (let i = 0; i < contour.length; i++) {
      const p1 = contour[(i - windowSize + contour.length) % contour.length];
      const p2 = contour[i];
      const p3 = contour[(i + windowSize) % contour.length];

      const angle = this.getAngle(p1, p2, p3);

      if (angle < angleThreshold) {
        corners.push(p2);
      }
    }

    // Use smaller merge distance to preserve more corners
    const mergeDistance = Math.max(5, Math.sqrt(area) * 0.06);
    return this.mergeNearbyCorners(corners, mergeDistance);
  }

  /**
   * Merges corners that are too close to each other.
   */
  private mergeNearbyCorners(corners: Point[], minDistance: number): Point[] {
    if (corners.length < 2) return corners;

    const merged: Point[] = [];
    const minDistanceSq = minDistance * minDistance;
    const toSkip = new Set<number>();

    for (let i = 0; i < corners.length; i++) {
      if (toSkip.has(i)) continue;

      const cluster: Point[] = [corners[i]];
      for (let j = i + 1; j < corners.length; j++) {
        if (toSkip.has(j)) continue;
        const distSq = (corners[i].x - corners[j].x) ** 2 + (corners[i].y - corners[j].y) ** 2;
        if (distSq < minDistanceSq) {
          cluster.push(corners[j]);
          toSkip.add(j);
        }
      }

      // Average the points in the cluster to get the new corner
      const avg = cluster.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
      avg.x /= cluster.length;
      avg.y /= cluster.length;
      merged.push(avg);
    }

    return merged;
  }

  /**
   * Calculates the angle between three points.
   */
  private getAngle(p1: Point, p2: Point, p3: Point): number {
    const dx1 = p1.x - p2.x;
    const dy1 = p1.y - p2.y;
    const dx2 = p3.x - p2.x;
    const dy2 = p3.y - p2.y;
    const dot = dx1 * dx2 + dy1 * dy2;
    const det = dx1 * dy2 - dy1 * dx2;
    return Math.abs(Math.atan2(det, dot));
  }

  /**
   * Simplify contour using Douglas-Peucker algorithm
   */
  private simplifyContour(points: Point[], epsilon: number): Point[] {
    if (points.length < 3) return points;

    // Find the point with maximum distance from line between first and last
    let maxDist = 0;
    let maxIndex = 0;

    const first = points[0];
    const last = points[points.length - 1];

    for (let i = 1; i < points.length - 1; i++) {
      const dist = this.perpendicularDistance(points[i], first, last);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }

    // If max distance is greater than epsilon, recursively simplify
    if (maxDist > epsilon) {
      const left = this.simplifyContour(points.slice(0, maxIndex + 1), epsilon);
      const right = this.simplifyContour(points.slice(maxIndex), epsilon);

      return [...left.slice(0, -1), ...right];
    } else {
      return [first, last];
    }
  }

  /**
   * Calculate perpendicular distance from point to line
   */
  private perpendicularDistance(
    point: Point,
    lineStart: Point,
    lineEnd: Point
  ): number {
    const dx = lineEnd.x - lineStart.x;
    const dy = lineEnd.y - lineStart.y;

    const numerator = Math.abs(
      dy * point.x - dx * point.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x
    );
    const denominator = Math.sqrt(dx * dx + dy * dy);

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Analyze geometric properties
   */
  private analyzeGeometry(
    polygon: Point[],
    area: number,
    perimeter: number,
    detectedCorners: number
  ): {
    vertices: number;
    circularity: number;
    aspectRatio: number;
    isConvex: boolean;
    solidity: number;
    width: number;
    height: number;
  } {
    // Use detected corners count (more reliable than polygon vertices from Douglas-Peucker)
    const vertices = detectedCorners;

    // Circularity: 4π × Area / Perimeter²
    // Circle = 1.0, square ≈ 0.785
    const circularity =
      perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;

    // Aspect ratio and solidity from bounding box
    const xs = polygon.map((p) => p.x);
    const ys = polygon.map((p) => p.y);
    const width = Math.max(...xs) - Math.min(...xs) + 1;
    const height = Math.max(...ys) - Math.min(...ys) + 1;
    const aspectRatio = height > 0 ? width / height : 1;
    const solidity = width * height > 0 ? area / (width * height) : 0;

    // Check convexity
    const isConvex = this.isConvexPolygon(polygon);

    return { vertices, circularity, aspectRatio, isConvex, solidity, width, height };
  }

  /**
   * Check if polygon is convex
   */
  private isConvexPolygon(points: Point[]): boolean {
    if (points.length < 3) return true;

    let sign = 0;

    for (let i = 0; i < points.length; i++) {
      const p1 = points[i];
      const p2 = points[(i + 1) % points.length];
      const p3 = points[(i + 2) % points.length];

      const crossProduct =
        (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x);

      if (crossProduct !== 0) {
        const currentSign = crossProduct > 0 ? 1 : -1;
        if (sign === 0) {
          sign = currentSign;
        } else if (sign !== currentSign) {
          return false; // Not convex
        }
      }
    }

    return true;
  }

  /**
   * Robust shape classification using multiple geometric features
   * Designed to handle various test scenarios: simple shapes, rotated, noisy backgrounds, etc.
   */
  private classifyShape(features: {
    vertices: number;
    circularity: number;
    aspectRatio: number;
    isConvex: boolean;
    solidity: number;
  }): {
    type:
      | "circle"
      | "triangle"
      | "rectangle"
      | "pentagon"
      | "star"
      | "line"
      | null;
    confidence: number;
  } {
    const { vertices, circularity, isConvex, aspectRatio, solidity } = features;

    // Line detection - extreme aspect ratio with low solidity
    if ((aspectRatio > 8 || aspectRatio < 0.125) && solidity < 0.6) {
      return { type: "line", confidence: 0.95 };
    }

    // Star detection - non-convex is the key indicator
    if (!isConvex && vertices >= 5) {
      // Additional check: stars have moderate to low circularity
      if (circularity < 0.7) {
        return { type: "star", confidence: 0.85 };
      }
    }

    // Use vertex count as PRIMARY classifier when in reliable range (3-6)
    // This prevents misclassification even when other features are ambiguous
    if (vertices >= 3 && vertices <= 6) {
      // Triangle: exactly 3 vertices
      if (vertices === 3) {
        return { type: "triangle", confidence: 0.92 };
      }

      // Rectangle: 4 vertices
      if (vertices === 4 && solidity > 0.6) {
        return { type: "rectangle", confidence: 0.95 };
      }

      // Pentagon: 5 vertices (regardless of convexity issues from pixel noise)
      if (vertices === 5) {
        // Pentagon: high circularity + aspect ratio very close to 1 + HIGH solidity
        if (circularity > 0.7 && aspectRatio > 0.92 && aspectRatio < 1.08 && solidity > 0.65) {
          return { type: "pentagon", confidence: 0.88 };
        }
        // Rectangle with noise: Low solidity (rotated rectangles have lower solidity)
        if (solidity < 0.60) {
          return { type: "rectangle", confidence: 0.85 };
        }
        // Rectangle with noise: elongated shape
        if (aspectRatio > 1.12 || aspectRatio < 0.88) {
          return { type: "rectangle", confidence: 0.85 };
        }
        // Default for ambiguous 5 vertices: assume rectangle (more common)
        return { type: "rectangle", confidence: 0.75 };
      }

      // Pentagon: 6 vertices (over-detected pentagon)
      if (vertices === 6 && circularity > 0.65 && aspectRatio > 0.9 && aspectRatio < 1.1) {
        return { type: "pentagon", confidence: 0.80 };
      }
    }

    // Circle detection AFTER vertex-based classification
    // Only for shapes with very few or no detected corners
    if (vertices <= 2 && aspectRatio > 0.8 && aspectRatio < 1.25) {
      return { type: "circle", confidence: 0.90 };
    }
    if (circularity > 0.85 && vertices <= 3 && aspectRatio > 0.75 && aspectRatio < 1.3) {
      return { type: "circle", confidence: 0.88 };
    }

    // Fallback: use shape characteristics when vertex count is not in reliable range
    if (isConvex && solidity > 0.7) {
      // Elongated shapes are likely rectangles
      if (aspectRatio > 1.4 || aspectRatio < 0.7) {
        return { type: "rectangle", confidence: 0.75 };
      }

      // Nearly square, convex, decent solidity - could be pentagon or rectangle
      if (aspectRatio > 0.9 && aspectRatio < 1.1) {
        // Pentagon has higher circularity than rectangles/squares
        if (circularity > 0.7) {
          return { type: "pentagon", confidence: 0.72 };
        }
        // Lower circularity suggests rectangle/square
        return { type: "rectangle", confidence: 0.70 };
      }

      // Default convex shape
      return { type: "rectangle", confidence: 0.65 };
    }

    // If nothing matches confidently, return null
    return { type: null, confidence: 0 };
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});

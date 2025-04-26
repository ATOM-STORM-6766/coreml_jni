/*
 * Copyright (C) AtomStorm.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.atomstorm.coreml;

import org.opencv.core.Point;
import org.opencv.core.Rect2d;

/**
 * JNI bindings for interacting with the native CoreML inference library.
 * Provides methods to load models, run inference, and manage native resources.
 */
public class CoreMLJNI {
    /**
     * Specifies the version of the YOLO model being used.
     * This affects how the raw detections are processed.
     */
    public static enum ModelVersion {
        /** You Only Look Once v5 */
        YOLO_V5,
        /** You Only Look Once v8 */
        YOLO_V8,
        /** You Only Look Once v11 (hypothetical or future version) */
        YOLO_V11 // TODO: Check if YOLOv11 is a standard or custom version.
    }

    /**
     * Specifies which processing units the CoreML model can utilize.
     */
    public static enum CoreMask {
        /** Use only the CPU. */
        CPU_ONLY,
        /** Use the CPU and the integrated GPU. */
        CPU_AND_GPU,
        /** Use all available processing units (CPU, GPU, Neural Engine). */
        ALL,
        /** Use the CPU and the Apple Neural Engine (ANE). */
        CPU_AND_NEURAL_ENGINE
    }

    /**
     * Represents a single detection result from the CoreML model.
     */
    public static class CoreMLResult {
        /**
         * Constructs a CoreMLResult.
         *
         * @param left The x-coordinate of the top-left corner of the bounding box.
         * @param top The y-coordinate of the top-left corner of the bounding box.
         * @param right The x-coordinate of the bottom-right corner of the bounding box.
         * @param bottom The y-coordinate of the bottom-right corner of the bounding box.
         * @param conf The confidence score of the detection (typically 0.0 to 1.0).
         * @param class_id The integer ID of the detected class.
         */
        public CoreMLResult(
            int left, int top, int right, int bottom, float conf, int class_id
        ) {
            this.conf = conf;
            this.class_id = class_id;
            this.rect = new Rect2d(new Point(left, top), new Point(right, bottom));
        }
        
        /** The bounding box of the detection. */
        public final Rect2d rect;
        /** The confidence score associated with this detection. */
        public final float conf;
        /** The identifier for the detected class. */
        public final int class_id;

        @Override
        public String toString() {
            return "CoreMLResult [rect=" + rect + ", conf=" + conf + ", class_id=" + class_id + "]";
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + ((rect == null) ? 0 : rect.hashCode());
            result = prime * result + Float.floatToIntBits(conf);
            result = prime * result + class_id;
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            CoreMLResult other = (CoreMLResult) obj;
            if (rect == null) {
                if (other.rect != null)
                    return false;
            } else if (!rect.equals(other.rect))
                return false;
            if (Float.floatToIntBits(conf) != Float.floatToIntBits(other.conf))
                return false;
            if (class_id != other.class_id)
                return false;
            return true;
        }
    }

    /**
     * Create a CoreML detector. Returns valid pointer on success, or NULL on error
     * @param modelPath Absolute path to the model on disk
     * @param numClasses How many classes. MUST MATCH or native code segfaults
     * @param modelVer Which model is being used. Detections will be incorrect if not set to corrresponding model.
     * @param coreMask Which compute unit to use.
     * @return Pointer to the detector in native memory
     */
    public static native long create(String modelPath, int numClasses, int modelVer, int coreMask);

    /**
     * Given an already running detector, change the bitmask controlling which
     * of the 3 cores the model is running on
     * @param ptr Pointer to detector in native memory
     * @param desiredCore Which of the three cores to operate on
     * @return return code of call, indicating success or failure
     */
    public static native int setCoreMask(long ptr, CoreMask desiredCore);
    
    /**
     * Delete all native resources assocated with a detector
     * @param ptr Pointer to detector in native memory
     */
    public static native void destroy(long ptr);

    /**
     * Run detction
     * @param detectorPtr Pointer to detector created above
     * @param imagePtr Pointer to a cv::Mat input image
     * @param nmsThresh Non-Maximum Suppression threshold
     * @param boxThresh Bounding box confidence threshold
     * @return Array of CoreMLResult objects containing the detection results
     */
    public static native CoreMLResult[] detect(
        long detectorPtr, long imagePtr, double nmsThresh, double boxThresh
    );
}

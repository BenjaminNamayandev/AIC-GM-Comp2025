#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include "net.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

const char *param_file = "/home/mrben/models/best-11s150_ncnn_model/model.ncnn.param";
const char *bin_file = "/home/mrben/models/best-11s150_ncnn_model/model.ncnn.bin";
const char *input_folder = "/home/mrben/data/test-thermal-data/test_images_8_bit"; // Folder containing JPEG images
const char *output_file = "/home/mrben/res.txt";                                   // Output results file

bool file_exists(const std::string &file_path)
{
    return std::filesystem::exists(file_path);
}

struct Object
{
    int class_id;
    float confidence;
    int min_x, min_y, max_x, max_y;

    Object(int id, float conf, int x1, int y1, int x2, int y2)
        : class_id(id), confidence(conf), min_x(x1), min_y(y1), max_x(x2), max_y(y2) {}
};

static float intersection_area(const Object &a, const Object &b)
{
    int overlap_x1 = std::max(a.min_x, b.min_x);
    int overlap_y1 = std::max(a.min_y, b.min_y);
    int overlap_x2 = std::min(a.max_x, b.max_x);
    int overlap_y2 = std::min(a.max_y, b.max_y);

    int width = std::max(overlap_x2 - overlap_x1, 0);
    int height = std::max(overlap_y2 - overlap_y1, 0);
    return width * height;
}

static void nms(std::vector<Object> &objects, float threshold)
{
    std::sort(objects.begin(), objects.end(),
              [](const Object &a, const Object &b)
              { return a.confidence > b.confidence; });

    for (size_t i = 0; i < objects.size(); i++)
    {
        auto &a = objects[i];
        if (a.confidence == 0.0f)
            continue;

        for (size_t j = i + 1; j < objects.size(); j++)
        {
            auto &b = objects[j];
            if (b.confidence == 0.0f)
                continue;

            float overlap = intersection_area(a, b);
            float area_a = (a.max_x - a.min_x) * (a.max_y - a.min_y);
            float area_b = (b.max_x - b.min_x) * (b.max_y - b.min_y);
            float iou = overlap / (area_a + area_b - overlap);

            if (iou > threshold)
            {
                b.confidence = 0.0f; // Mark for removal
            }
        }
    }

    // Remove low-confidence objects
    objects.erase(
        std::remove_if(objects.begin(), objects.end(),
                       [](const Object &obj)
                       { return obj.confidence == 0.0f; }),
        objects.end());
}

int main()
{

    std::cout << "YOU SHOULD SEE ME" << std::endl;

    // Check if model files exist
    if (!file_exists(param_file))
    {
        std::cerr << "Param file not found: " << param_file << std::endl;
        return -1;
    }

    if (!file_exists(bin_file))
    {
        std::cerr << "Bin file not found: " << bin_file << std::endl;
        return -1;
    }

    std::cout << "Both model files exist!" << std::endl;

    // Load YOLO model
    ncnn::Net net;
    net.opt.num_threads = 4; // Optimize for Raspberry Pi (4 cores)

    if (std::filesystem::exists(param_file))
    {
        std::cout << "File 1 exists" << std::endl;
    }
    else
    {
        std::cerr << "File 1 failed" << std::endl;
    }

    if (net.load_param(param_file) != 0 || net.load_model(bin_file) != 0)
    {
        std::cerr << "Failed to load NCNN model files." << std::endl;
        return -1;
    }

    // Open output file
    std::ofstream result_file(output_file);
    if (!result_file.is_open())
    {
        std::cerr << "Error: Cannot write to output file." << std::endl;
        return -1;
    }

    // Print input blob names
    const auto &input_names = net.input_names();
    for (const auto &name : input_names)
    {
        std::cout << "Input blob name: " << name << std::endl;
    }

    // Print output blob names
    const auto &output_names = net.output_names();
    for (const auto &name : output_names)
    {
        std::cout << "Output blob name: " << name << std::endl;
    }

    // Loop through all JPEG files in folder
    for (const auto &entry : fs::directory_iterator(input_folder))
    {
        if (entry.path().extension() != ".jpeg")
            continue;

        std::string file_path = entry.path().string();
        std::string file_name = entry.path().filename().string();

        // Load JPEG using stb_image
        int img_w, img_h, img_c;
        unsigned char *img_data = stbi_load(file_path.c_str(), &img_w, &img_h, &img_c, 1);
        if (!img_data)
        {
            std::cerr << "Failed to load image: " << file_path << std::endl;
            continue;
        }

        // Convert image to NCNN format (resize for YOLO input)
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_data, ncnn::Mat::PIXEL_GRAY, img_w, img_h, 640, 640);
        stbi_image_free(img_data); // Free memory after conversion

        // Run inference
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true); // Optimize for speed
        ex.input("in0", in);

        ncnn::Mat out;
        ex.extract("out0", out);

        // YOLO output format: [batch, num_proposals, 5 + num_classes]
        const int num_classes = 3; // Should match your model's classes
        const float confidence_threshold = 0.25f;
        const float nms_threshold = 0.45f;

        std::vector<Object> objects;
        const float *data = out.row(0); // Get first (and only) batch

        for (int i = 0; i < out.h; i++)
        { // Each row is a proposal
            const float *values = data + i * out.w;

            // Get objectness score
            float objectness = values[4];
            if (objectness < confidence_threshold)
                continue;

            // Find class with maximum probability
            int class_id = -1;
            float max_confidence = 0.0f;
            for (int c = 0; c < num_classes; c++)
            {
                float class_confidence = values[5 + c];
                if (class_confidence > max_confidence)
                {
                    max_confidence = class_confidence;
                    class_id = c;
                }
            }

            // Final confidence = objectness * class_confidence
            float confidence = objectness * max_confidence;
            if (confidence < confidence_threshold)
                continue;

            // Get bounding box (center-x, center-y, width, height format)
            float cx = values[0] * img_w / 640.0f;
            float cy = values[1] * img_h / 640.0f;
            float w = values[2] * img_w / 640.0f;
            float h = values[3] * img_h / 640.0f;

            // Convert to min/max coordinates
            int min_x = static_cast<int>(cx - w / 2);
            int min_y = static_cast<int>(cy - h / 2);
            int max_x = static_cast<int>(cx + w / 2);
            int max_y = static_cast<int>(cy + h / 2);

            // Store detection
            objects.emplace_back(class_id, confidence, min_x, min_y, max_x, max_y);
        }

        // Apply Non-Maximum Suppression (NMS)
        nms(objects, nms_threshold);

        // Write to file
        for (const auto &obj : objects)
        {
            result_file << file_name << " " << obj.class_id << " " << obj.confidence << " "
                        << obj.min_x << " " << obj.min_y << " " << obj.max_x << " " << obj.max_y << "\n";
        }

        result_file.close();
        std::cout << "Processing complete. Results saved to: " << output_file << std::endl;
        return 0;
    }
}
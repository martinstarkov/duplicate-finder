#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "opencv2/core/base.hpp"
#include "opencv2/imgproc/types_c.h"

#include <nlohmann/json.hpp>

#include <bit>
#include <cassert>
#include <array>
#include <iostream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <bitset>

using json = nlohmann::json;
namespace fs = std::filesystem;

// TODO: Figure out why hash for DHash and AHash is negative.

#define LOG(x) { std::cout << x << std::endl; }



struct File {
    File(const cv::Mat& image, const fs::path& path, bool flag) : 
    image{ image }, original_image{ image }, path{ path }, flag{ flag } {}
    cv::Mat image;
    cv::Mat original_image;
    fs::path path;
    bool flag;
};

std::uint64_t GetDHash(const cv::Mat& resized_image) {
    assert(resized_image.rows == resized_image.cols - 1);
    std::uint64_t hash{ 0 };
    int i{ 0 }; // counts the current index in the hash
    // Cycle through every row 
    for (int y = 0; y < resized_image.rows; ++y) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + y * resized_image.step);
        // Cycle through every column
        for (int x = 0; x < resized_image.cols - 1; ++x) {
            // If the next pixel is brighter, make the hash contain a 1, else keep it as 0
            if (ptr[x + 1] > ptr[x]) {
                hash |= (std::uint64_t)1 << i;
            }
            i++; // increment hash index
        }
    }
    assert(hash != 0);
    return hash;
}

inline std::uint64_t GetLinearIndex(std::size_t hash_count, std::uint64_t i, std::uint64_t j) {
    return hash_count * (hash_count - 1) / 2 - (hash_count - j) * ((hash_count - j) - 1) / 2 + i - j - 1;
}


void AddBorder(cv::Mat& image, int size, const cv::Scalar& color) {
    cv::Rect inner_rectangle{ size, size, image.cols - size * 2, image.rows - size * 2 };
    cv::Mat border;
    border.create(image.rows, image.cols, image.type());
    border.setTo(color);
    image(inner_rectangle).copyTo(border(inner_rectangle));
    image = border;
}

cv::Mat ToGreyscale(const cv::Mat& image) {
    cv::Mat greyscale;
    cv::cvtColor(image, greyscale, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    return greyscale;
}

inline std::uint64_t GetHammingDistance(std::uint64_t hash1, std::uint64_t hash2) {
    return std::popcount(hash1 ^ hash2); // XOR (get number of 1 bits in the difference of 1 and 2).
}

std::vector<std::uint8_t> ProcessDuplicates(std::vector<std::uint64_t>& hashes, int hamming_threshold) {
    std::size_t hash_count{ hashes.size() };
    LOG("[Starting duplicate search through " << hash_count << " hashes]");

    std::size_t pair_count{ hash_count * (hash_count - 1) / 2 };
    
    std::vector<std::uint8_t> pairs;
    pairs.resize(pair_count, 0);
    
    for (auto i{ 0 }; i < hash_count; ++i) {
        for (auto j{ 0 }; j < i; ++j) {
            auto index = GetLinearIndex(hash_count, i, j);
            auto hamming_distance{ GetHammingDistance(hashes[i], hashes[j]) };
            if (hamming_distance <= hamming_threshold) {
                //int similarity{ static_cast<int>(1.0 - static_cast<double>(hamming_distance) / (static_cast<double>(hamming_threshold) + 1.0) * 100.0) };
                pairs[index] = hamming_distance + 1;
            }
        }
    }
    LOG("[Finished duplicate search]");
    return pairs;
}

// @arg parent_directories Vector of pair elements: [directory path, recursive_search]
std::vector<fs::path> GetMediaFiles(const std::vector<std::pair<std::string, bool>>& parent_directories) {
    LOG("[Starting media file search through " << parent_directories.size() << " parent directory(ies)]");
    std::vector<fs::path> files;
    auto add_if_regular_file = [&](const fs::directory_entry& file) {
        if (file.is_regular_file())
            files.emplace_back(fs::absolute(file.path()));
    };
    for (auto& [directory_path, recursive_search] : parent_directories)
        if (recursive_search)
            for (const fs::directory_entry& file : fs::recursive_directory_iterator{ directory_path })
                add_if_regular_file(file);
        else
            for (const fs::directory_entry& file : fs::directory_iterator{ directory_path })
                add_if_regular_file(file);
    LOG("[Finished media file search: " << files.size() << " media file(s) found]");
    return files;
}


std::uint64_t GetAHash(const cv::Mat& resized_image) {
    std::uint64_t hash{ 0 };
    int i{ 0 }; // counts the current index in the hash
    // Cycle through every row
    std::size_t sum{ 0 };
    std::size_t pixels{ static_cast<std::size_t>(resized_image.rows* resized_image.cols) };
    // Find average of all pixel brightnesses
    for (int index = 0; index < pixels; ++index) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + index);
        sum += *ptr;
    }
    std::size_t average{ sum / pixels };
    for (int y = 0; y < resized_image.rows; ++y) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + y * resized_image.step);
        // Cycle through every column
        for (int x = 0; x < resized_image.cols; ++x) {
            if (ptr[x] > average) {
                hash |= (std::uint64_t)1 << i;
            }
            i++; // increment hash index
        }
    }
    assert(hash != 0);
    return hash;
}

cv::Mat ResizeImage(const cv::Mat& image, int width, int height) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(8, 8), 0, 0, cv::INTER_AREA);
    return resized;
}

enum class FileType {
    UNKNOWN = -1,
    PICTURE = 0,
    VIDEO = 1,
    HEIC = 2
};

FileType GetMediaType(const fs::path& extension) {
    if      (extension.compare(".JPG")  == 0 || 
             extension.compare(".PNG")  == 0 ||
             extension.compare(".JPEG") == 0 ||
             extension.compare(".WEBP") == 0 ||
             extension.compare(".BMP")  == 0) return FileType::PICTURE;
    else if (extension.compare(".HEIC") == 0 ||
             extension.compare(".HEIF") == 0) return FileType::HEIC;
    else if (extension.compare(".MOV")  == 0 ||
             extension.compare(".MP4")  == 0 ||
             //extension.compare(".WMV")  == 0 ||
             extension.compare(".GIF")  == 0 ||
             extension.compare(".AVI")  == 0) return FileType::VIDEO;
    else return FileType::UNKNOWN;
}

cv::Mat GetVideoFirstFrame(const fs::path& path, bool greyscale) {
    cv::VideoCapture cap{ path.generic_string(), cv::CAP_FFMPEG };
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    cv::Mat first_frame;
    if (cap.read(first_frame))
        if (greyscale)
            first_frame = ToGreyscale(first_frame);
    return first_frame;
}

cv::Mat GetImage(const fs::path& path, bool greyscale) {
    std::string extension{ path.extension().string() };
    std::transform(extension.begin(), extension.end(), extension.begin(), ::toupper);
    cv::Mat image;
    switch (GetMediaType(extension)) {
        case FileType::PICTURE:
        {
            if (greyscale)
                image = cv::imread(path.generic_string(), cv::IMREAD_GRAYSCALE);
            else
                image = cv::imread(path.generic_string());
            break;
        }
        case FileType::HEIC:
        {
            break;
        }
        case FileType::VIDEO:
        {
            image = GetVideoFirstFrame(path, greyscale);
            break;
        }
        case FileType::UNKNOWN:
        {
            break;
        }
    }
    return image;
}

void ComputePHash(cv::Mat input, cv::OutputArray outputArr) {
    CV_Assert(input.type() == CV_8UC4 ||
              input.type() == CV_8UC3 ||
              input.type() == CV_8U);

    cv::Mat bitsImg;
    cv::Mat grayImg;
    cv::Mat resizeImg;
    cv::Mat dctImg;
    cv::Mat grayFImg;
    cv::Mat topLeftDCT;
    cv::resize(input, resizeImg, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR_EXACT);
    if (input.channels() > 1)
        cv::cvtColor(resizeImg, grayImg, cv::COLOR_BGR2GRAY);
    else
        grayImg = resizeImg;

    grayImg.convertTo(grayFImg, CV_32F);
    cv::dct(grayFImg, dctImg);
    dctImg(cv::Rect(0, 0, 8, 8)).copyTo(topLeftDCT);
    topLeftDCT.at<float>(0, 0) = 0;
    float const imgMean = static_cast<float>(cv::mean(topLeftDCT)[0]);

    cv::compare(topLeftDCT, imgMean, bitsImg, cv::CMP_GT);
    bitsImg /= 255;
    outputArr.create(1, 8, CV_8U);
    cv::Mat hash = outputArr.getMat();
    uchar* hash_ptr = hash.ptr<uchar>(0);
    uchar const* bits_ptr = bitsImg.ptr<uchar>(0);
    std::bitset<8> bits;
    for (size_t i = 0, j = 0; i != bitsImg.total(); ++j) {
        for (size_t k = 0; k != 8; ++k) {
            //avoid warning C4800, casting do not work
            bits[k] = bits_ptr[i++] != 0;
        }
        hash_ptr[j] = static_cast<uchar>(bits.to_ulong());
    }
}

void ComputeAHash(cv::InputArray inputArr, cv::OutputArray outputArr) {
    cv::Mat const input = inputArr.getMat();
    CV_Assert(input.type() == CV_8UC4 ||
              input.type() == CV_8UC3 ||
              input.type() == CV_8U);

    cv::Mat bitsImg;
    cv::Mat grayImg;
    cv::Mat resizeImg;
    cv::resize(input, resizeImg, cv::Size(8, 8), 0, 0, cv::INTER_LINEAR_EXACT);
    if (input.channels() > 1)
        cv::cvtColor(resizeImg, grayImg, cv::COLOR_BGR2GRAY);
    else
        grayImg = resizeImg;

    uchar const imgMean = static_cast<uchar>(cvRound(cv::mean(grayImg)[0]));
    cv::compare(grayImg, imgMean, bitsImg, cv::CMP_GT);
    bitsImg /= 255;
    outputArr.create(1, 8, CV_8U);
    cv::Mat hash = outputArr.getMat();
    uchar* hash_ptr = hash.ptr<uchar>(0);
    uchar const* bits_ptr = bitsImg.ptr<uchar>(0);
    std::bitset<8> bits;
    for (size_t i = 0, j = 0; i != bitsImg.total(); ++j) {
        for (size_t k = 0; k != 8; ++k) {
            //avoid warning C4800, casting do not work
            bits[k] = bits_ptr[i++] != 0;
        }
        hash_ptr[j] = static_cast<uchar>(bits.to_ulong());
    }
}

bool HashExists(const std::vector<std::tuple<std::string, std::string, double>>& hashes,
               const std::string& pathA, const std::string& pathB) {
    for (const auto& [p1, p2, s] : hashes) {
        if ((p1 == pathA && p2 == pathB) || (p1 == pathB && p2 == pathA)) return true;
    }
    return false;
}

json GetMediaHashes(const std::vector<fs::path>& files, std::string output_file) {
    std::size_t file_count{ files.size() };

    LOG("[Starting hashing of " << file_count << " media file(s)]");

    //std::unordered_map<std::string, std::vector<std::string>> hashes;
    std::vector<std::tuple<std::string, std::string, double>> hashes;
    hashes.reserve(file_count);

    //json json_hashes;
    //std::string json_filepath{ "path_hashes_1.json" };

    int downscale_size{ 16 };

    std::unordered_map<std::string, cv::Mat> individual_hashes;
    individual_hashes.reserve(file_count);

    double similarity_threshold{ 20 };

    for (auto i{ 0 }; i < file_count; ++i) {
        const auto& fileA{ files[i] };
        std::string pathA{ fileA.generic_string() };
        if (pathA.find("/") != std::string::npos && fs::exists(pathA)) {
            cv::Mat imageA{ GetImage(fileA, false) };
            if (!imageA.empty()) {
                cv::Mat hashA;
                ComputePHash(imageA, hashA);
                individual_hashes[pathA] = hashA;
            }
        }
        if (i % 100 == 0) LOG("[Hashing [" << i << "/" << file_count << "] media file(s)...]");
    }

    for (auto i{ 0 }; i < file_count; ++i) {
        const auto& fileA{ files[i] };
        std::string pathA{ fileA.generic_string() };
        if (pathA.find("/") != std::string::npos && fs::exists(pathA)) {
            for (auto j{ 0 }; j < file_count; ++j) {
                const auto& fileB{ files[j] };
                std::string pathB{ fileB.generic_string() };
                if (pathB.find("/") != std::string::npos && fs::exists(pathB) && pathA != pathB) {
                    auto itA{ individual_hashes.find(pathA) };
                    if (itA != individual_hashes.end()) {
                        auto itB{ individual_hashes.find(pathB) };
                        if (itB != individual_hashes.end()) {
                            double similarity{ cv::norm(itA->second, itB->second, cv::NORM_HAMMING) };
                            if (similarity < similarity_threshold && !HashExists(hashes, pathA, pathB)) {
                                hashes.emplace_back(pathA, pathB, similarity);
                            }
                        }
                    }
                }
            }
        }
        if (i % 100 == 0) LOG("[Comparing [" << i << "/" << file_count << "] media file(s)...]");
    }

    
    json json_hashes{ hashes };
    std::ofstream o(output_file);
    try {
        o << std::setw(4) << json_hashes << std::endl;
    } catch (...) {
        LOG("[Failed to add hashes to json file]");
    }

    /*for (auto& [path, hash] : hashes) {
        try {
            if (path != "<Invalid characters in string.>" &&
                path != "Invalid characters in string." &&
                path != "Invalid characters in string") {
                json_hashes[path] = hash;
            }
        } catch (...) {
            LOG("[Failed to add hash: " << hash << " to json file]");
            continue;
        }
    }*/

    //if (!image.empty()) {
    //LOG("Hashing: " << file.generic_string());
    //cv::Mat hash;

    //std::stringstream h;

    /*for (int i = 0; i < hash.cols; i++)
        h << std::hex << std::int64_t(hash.at<uchar>(0, i)) << "\t";
    std::int64_t x;
    h >> x;*/
    //std::uint64_t hash{ GetAHash(thumbnail) };
    //hashes[path] = x;
    /*std::string str_hash{ std::to_string(hash) };
    auto it{ hashes.find(str_hash) };
    if (it == hashes.end())
        hashes[str_hash] = std::vector<std::string>{ path };
    else
        it->second.emplace_back(path);*/
        //std::ofstream o(json_filepath);
        //o << std::setw(4) << json_hashes << std::endl;
        //LOG(i << ": " << hashes[i]);
    //}

    /*for (auto& [hash, vector] : hashes) {
        try {
            json_hashes[hash] = vector;
        } catch (...) {
            LOG("[Failed to add hash: " << hash << " to json file]");
            continue;
        }
    }*/

    /*std::string json_filepath{ "FILE_HASHES_ALL_INCLPRIV_16.json" };

    try {
        std::ofstream o(json_filepath);
        o << std::setw(4) << json_hashes << std::endl;
    } catch (...) {
        LOG("[Failed to load json_hashes to json file");
    }
    hashes;*/
    LOG("[Successfully wrote all media file hashes to " << output_file << "]");
    return json_hashes;
}




int main(int argc, char** argv) {
    //cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    /*std::fstream file{ "C:/Users/Martin/Desktop/hashes_3.csv" };

    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
    std::string line, word;

    std::vector<std::tuple<std::string, std::string, double>> hashes;
    std::size_t current_row{ 0 };*/

    //std::unordered_map<std::string, std::vector<std::string>> hashes;


    //if (file.is_open()) {
    //    while (std::getline(file, line)) {
    //        ++current_row;
    //        std::stringstream str(line);

    //        std::string original{ str.str() };

    //        
    //        std::string s = original.substr(4, original.length() - 4);
    //        if (s.find("/") == std::string::npos) {
    //            std::cout << "Failed on row: " << current_row << std::endl;
    //        } else {
    //            std::string file = s.substr(0, s.find('\"\", '));
    //            std::string s2 = s.substr(s.find('\"\", ') + 6, s.length() - s.find('\"\", ') - 8);
    //            std::string file2 = s2.substr(0, s2.find('\"\", '));
    //            std::string hash_str = s2.substr(s2.find('\"\", ') + 4, s2.length() - s2.find('\"\", '));
    //            std::stringstream sstream(hash_str);
    //            double hash{ 0 };
    //            sstream >> hash;
    //            hashes.emplace_back(file, file2, hash);
    //            //std::cout << original << " -> " << file << ", " << hash_str << std::endl;
    //            //std::stringstream sstream(hash_str);
    //            //std::uint64_t hash;
    //            //sstream >> hash;

    //            /*auto it{ hashes.find(hash_str) };
    //            if (it == hashes.end())
    //                hashes[hash_str] = std::vector<std::string>{ file };
    //            else
    //                it->second.emplace_back(file);*/
    //        }
    //    }
    //} else
    //    std::cout << "Could not open the file\n";


    //std::size_t original_count = hashes.size();

   /* const auto removed = std::erase_if(hashes, [](const auto& item) {
        auto const& [key, value] = item;
        return value.size() <= 1;
    });

    std::size_t duplicate_count = original_count - removed;*/

   /* LOG("[Found " << original_count << " duplicates]");

    try {
        std::ofstream o("C:/Users/Martin/Desktop/duplicate_finder/output_test_3.json");
        json json_hashes{ hashes };
        o << std::setw(4) << json_hashes << std::endl;
    } catch (...) {
        LOG("[Failed to load json_hashes to json file");
    }*/
    //std::ifstream f("example.json");
    //json data = json::parse(f);
    // TODO: Figure out Premature end of JPEG file
    //auto files{ GetMediaFiles({ "../vizsla_154/"}) };//, "../maltese_252/", "../vizsla_4048/" }) };
    //auto files{ GetMediaFiles({ "../test/" }) };
    auto files{ GetMediaFiles({ { "F:/Nicole/", true }, { "F:/Media/", true } }) };
    //auto files{ GetMediaFiles({ { "F:/Media/", true } }) };
    //auto files{ GetMediaFiles({ { "F:/Nicole/", true } }) };
    //auto files{ GetMediaFiles({ { "C:/Dev/duplicate-finder/duplicates/JPG/", true } }) };
    auto hashes{ GetMediaHashes(files, "C:/Users/Martin/Desktop/duplicate_finder/output_while_martin_sleeps.json")};
    //auto pairs{ ProcessDuplicates(hashes, 0) };

    /*
    cv::Size window{ 800, 400 };
    auto hash_count{ hashes.size() };

    LOG("[Displaying duplicate pairs]");

    for (auto i{ 0 }; i < hash_count; ++i) {
        for (auto j{ 0 }; j < i; ++j) {
            auto index = GetLinearIndex(hash_count, i, j);
            auto hamming_distance = pairs[index] - 1;
            if (hamming_distance >= 0) {
                assert(i < files.size());
                assert(j < files.size());
                auto image1 = GetImage(files[i]);
                auto image2 = GetImage(files[i]);
                if (!image1.empty() && !image2.empty()) {
                    LOG("[" << (int)(index / pairs.size() * 100) << "% of files displayed]");
                    std::array<File, 2> images{
                        File(Resize(image1, window.width / 2, window.height), files[i], false),
                        File(Resize(image2, window.width / 2, window.height), files[j], true),
                    };

                    AddBorder(images[0].image, 10, { 0, 255, 0 });
                
                    cv::Mat concatenated;
                    // Concatenate duplicates images into one big image
                    std::array<cv::Mat, 2> matrixes{ images[0].image, images[1].image };
                    cv::hconcat(matrixes.data(), 2, concatenated);
                    cv::imshow("Duplicate Finder", concatenated);
                    cv::waitKey(0);
                }
            }
        }
    }

    LOG("[Finished displaying duplicate pairs]");

    cv::destroyAllWindows();
    */
}

#include <algorithm>
#include <limits>

#ifndef STENCIL_UTILS_H
#define STENCIL_UTILS_H
namespace stencil {
    constexpr auto NumChannels = 4u; // RGBA

    typedef struct {
        unsigned width;
        unsigned height;
        std::vector<unsigned char> bytes; // Note: this is in row, col, channel order
    } Image;

    typedef struct {
        unsigned width;
        unsigned height;
        unsigned origChanMin[NumChannels]; // Note the original image's min and max values per chan so that we can reconstruct an image with the same brightness
        unsigned origChanMax[NumChannels];
    } ImageDescriptor;

    auto loadPng(const std::string &filename) -> std::optional<Image> {
        auto img = Image{};

        unsigned error = lodepng::decode(img.bytes, img.width, img.height, filename);

        if (error) {
            std::cerr << "PNG decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
            return nullopt;
        }

        return img;
    }

    auto savePng(const Image &image, const std::string &filename) -> bool {
        unsigned error = lodepng::encode(filename, image.bytes, image.width, image.height);

        if (error) std::cerr << "PNG encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return error == 0;
    }

// buffer should be pre-allocated to the right size
    auto toPaddedFloatImage(const Image &image, const std::unique_ptr<float[]> &buffer) -> ImageDescriptor {
        auto fImg = ImageDescriptor{
                .width= image.width + 2,
                .height= image.height + 2,
                .origChanMin = {0, 0, 0, 0},
                .origChanMax = {0, 0, 0, 0},
        };
        unsigned char max[4] = {0, 0, 0, 0};
        constexpr auto MaxUChar = std::numeric_limits<unsigned char>::max();
        unsigned char min[4] = {MaxUChar, MaxUChar, MaxUChar, MaxUChar};
        for (auto idx = 0u; idx < image.width * image.height; idx++) {
            const auto rgba = &image.bytes[idx * NumChannels];
            for (auto chan = 0u; chan < NumChannels; chan++) {
                max[chan] = std::max(max[chan], rgba[chan]);
                min[chan] = std::min(min[chan], rgba[chan]);
            }
        }
        for (auto c = 0u; c < NumChannels; c++) {
            fImg.origChanMin[c] = min[c];
            fImg.origChanMax[c] = max[c];
            // copy the bytes to the middle
            for (auto y = 0u; y < image.height; y++) {
                for (auto x = 0u; x < image.width; x++) {
                    const auto cIdx = (y * image.width + x) * NumChannels + c;
                    const auto fIdx = c * (fImg.width * fImg.height) + (fImg.width * (y + 1)) + (x + 1);
                    buffer[fIdx] = (max[c] == min[c])
                                   ? 0.f :
                                   ((float) image.bytes[cIdx] - (float) min[c]) / ((float) max[c] - (float) min[c]);
                }
            }
            //zero out the top and bottom
            for (auto y = 1u; y < fImg.height - 1; y++) {
                const auto leftIdx = c * (fImg.width * fImg.height) + (fImg.width * y);
                const auto rightIdx = leftIdx + fImg.width - 1;
                buffer[leftIdx] = 0.f;
                buffer[rightIdx] = 0.f;
            }
            //zero out the left and right
            for (auto x = 0u; x < fImg.width; x++) {
                const auto topIdx = c * (fImg.width * fImg.height) + x;
                const auto bottomIdx = c * (fImg.width * fImg.height) + (fImg.width * (fImg.height - 1)) + x;
                buffer[topIdx] = 0.f;
                buffer[bottomIdx] = 0.f;
            }
        }

        return fImg;
    }

    auto toUnpaddedCharsImage(const ImageDescriptor &floatImage, const std::unique_ptr<float[]> &buffer) -> Image {
        auto img = Image{
                .width= floatImage.width - 2,
                .height= floatImage.height - 2,
                .bytes= std::vector<unsigned char>(NumChannels * (floatImage.height - 2) * (floatImage.width - 2), 0)
        };

        // Find the min and max vals of each channel
        float max[4] = {0.f, 0.f, 0.f, 0.f};
        constexpr auto MaxFloat = std::numeric_limits<float>::max();
        float min[4] = {MaxFloat, MaxFloat, MaxFloat, MaxFloat};

        for (auto c = 0u; c < NumChannels; c++) {
            for (auto y = 1u; y < floatImage.height - 1; y++) {
                for (auto x = 1u; x < floatImage.width - 1; x++) {
                    const auto idx = c * (floatImage.height * floatImage.width) + y * floatImage.width + x;
                    min[c] = std::min(min[c], buffer[idx]);
                    max[c] = std::max(max[c], buffer[idx]);
                }
            }
        }
        for (auto c = 0u; c < NumChannels; c++) {
            for (auto y = 0u; y < img.height; y++) {
                for (auto x = 0u; x < img.width; x++) {
                    const auto inIdx =
                            c * (floatImage.height * floatImage.width) + (y + 1) * floatImage.width + (x + 1);
                    const auto outIdx = (y * img.width + x) * NumChannels + c;

                    auto rescaled = (max[c] == min[c]) ? 0.f :
                                    (buffer[inIdx] + min[c]) /
                                    (max[c] - min[c]); // Now it's in the range 0..1
                    auto inOrigBrightness =
                            (rescaled * (float) (floatImage.origChanMax[c] - floatImage.origChanMin[c])) +
                            (float) floatImage.origChanMin[c];
                    img.bytes[outIdx] = (unsigned char) inOrigBrightness;
                }
            }
        }

        return img;
    }

}
#endif
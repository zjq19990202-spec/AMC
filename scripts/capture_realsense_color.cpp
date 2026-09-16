#include <librealsense2/rs.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: capture_realsense_color SERIAL OUTPUT.png\n";
    return 2;
  }
  rs2::config config;
  config.enable_device(argv[1]);
  config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
  rs2::pipeline pipeline;
  pipeline.start(config);
  rs2::frameset frames;
  // Let D435 auto-exposure and auto-white-balance settle before capture.
  for (int i = 0; i < 600; ++i) frames = pipeline.wait_for_frames();
  const auto color = frames.get_color_frame();
  if (!color) return 3;
  cv::Mat rgb(color.get_height(), color.get_width(), CV_8UC3,
              const_cast<void*>(color.get_data()), cv::Mat::AUTO_STEP);
  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  if (!cv::imwrite(argv[2], bgr)) return 4;
  std::cout << color.get_width() << "x" << color.get_height() << "\n";
  return 0;
}

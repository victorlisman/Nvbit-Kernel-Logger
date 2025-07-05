#include <gtest/gtest.h>
#include <fstream>
#include <regex>
#include <string>

class ArgumentDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Run write with NVBit and capture output
        system("LD_PRELOAD=../../tools/log_kernel_launch/log_kernel_launch.so ./write.elf > nvbit_output.txt 2>&1");
    }
    
    std::string readNVBitOutput() {
        std::ifstream file("nvbit_output.txt");
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        return content;
    }
};

TEST_F(ArgumentDetectionTest, DebugOutput) {
    std::string output = readNVBitOutput();
    std::cout << "=== ACTUAL NVBIT OUTPUT ===" << std::endl;
    std::cout << output << std::endl;
    std::cout << "=== END OUTPUT ===" << std::endl;
    EXPECT_TRUE(true); // Always pass, just for debugging
}

TEST_F(ArgumentDetectionTest, OutputArgumentDetected) {
    std::string output = readNVBitOutput();
    
    // Test Arg 0 - output array (based on actual format)
    std::regex arg0Regex(R"(\s+\[output\] Arg 0 \(float\*\): (0x[0-9a-f]+), \[0\] = ([0-9.-]+))");
    std::smatch arg0Match;
    ASSERT_TRUE(std::regex_search(output, arg0Match, arg0Regex))
        << "Arg 0 output detection failed";
    
    // Verify it's detected as output type
    EXPECT_TRUE(output.find("[output] Arg 0") != std::string::npos);
    
    // Verify output pointer is also reported
    std::regex outputPtrRegex(R"(\s+Output pointer: (0x[0-9a-f]+))");
    std::smatch outputPtrMatch;
    ASSERT_TRUE(std::regex_search(output, outputPtrMatch, outputPtrRegex))
        << "Output pointer not reported";
    
    // Verify output pointer matches Arg 0 address
    EXPECT_EQ(arg0Match[1].str(), outputPtrMatch[1].str())
        << "Output pointer doesn't match Arg 0 address";
}

TEST_F(ArgumentDetectionTest, KernelLaunchDetected) {
    std::string output = readNVBitOutput();
    
    // Check kernel name
    EXPECT_TRUE(output.find("Intercepted kernel launch: write(float*)") != std::string::npos)
        << "Kernel launch not detected";
    
    // Check grid dimensions
    std::regex gridRegex(R"(gridDim = \((\d+), (\d+), (\d+)\))");
    std::smatch gridMatch;
    ASSERT_TRUE(std::regex_search(output, gridMatch, gridRegex));
    
    EXPECT_EQ(std::stoi(gridMatch[1]), 4);   // gridDim.x
    EXPECT_EQ(std::stoi(gridMatch[2]), 1);   // gridDim.y
    EXPECT_EQ(std::stoi(gridMatch[3]), 1);   // gridDim.z
    
    // Check block dimensions
    std::regex blockRegex(R"(blockDim = \((\d+), (\d+), (\d+)\))");
    std::smatch blockMatch;
    ASSERT_TRUE(std::regex_search(output, blockMatch, blockRegex));
    
    EXPECT_EQ(std::stoi(blockMatch[1]), 128); // blockDim.x
    EXPECT_EQ(std::stoi(blockMatch[2]), 1);   // blockDim.y
    EXPECT_EQ(std::stoi(blockMatch[3]), 1);   // blockDim.z
}

TEST_F(ArgumentDetectionTest, ArgumentTypesCorrect) {
    std::string output = readNVBitOutput();
    
    // Count argument types
    size_t outputCount = 0;
    
    std::regex outputRegex(R"(\[output\] Arg)");
    
    std::sregex_iterator outputIter(output.begin(), output.end(), outputRegex);
    std::sregex_iterator end;
    
    outputCount = std::distance(outputIter, end);
    
    // Verify expected counts for write(float* out)
    EXPECT_EQ(outputCount, 1) << "Should have 1 output argument";
}

TEST_F(ArgumentDetectionTest, ArgumentDataTypesCorrect) {
    std::string output = readNVBitOutput();
    
    // Verify pointer argument is float*
    EXPECT_TRUE(output.find("Arg 0 (float*)") != std::string::npos);
}

TEST_F(ArgumentDetectionTest, NoInputOrScalarArgs) {
    std::string output = readNVBitOutput();
    
    // Verify no input or scalar arguments for this kernel
    EXPECT_TRUE(output.find("[input]") == std::string::npos)
        << "Should have no input arguments";
    EXPECT_TRUE(output.find("[scalar]") == std::string::npos) 
        << "Should have no scalar arguments";
}
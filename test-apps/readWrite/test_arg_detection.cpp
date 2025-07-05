#include <gtest/gtest.h>
#include <fstream>
#include <regex>
#include <string>

class ArgumentDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        system("LD_PRELOAD=../../tools/log_kernel_launch/log_kernel_launch.so ./readWrite.elf > nvbit_output.txt 2>&1");
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
    EXPECT_TRUE(true); 
}

TEST_F(ArgumentDetectionTest, InputArgumentDetected) {
    std::string output = readNVBitOutput();
    
    std::regex arg0Regex(R"(\s+\[input\] Arg 0 \(float\*\): (0x[0-9a-f]+), \[0\] = ([0-9.-]+))");
    std::smatch arg0Match;
    ASSERT_TRUE(std::regex_search(output, arg0Match, arg0Regex)) 
        << "Arg 0 input detection failed";
    
    EXPECT_TRUE(output.find("[input] Arg 0") != std::string::npos);
    
    EXPECT_EQ(std::stof(arg0Match[2].str()), 0.0f);
}

TEST_F(ArgumentDetectionTest, OutputArgumentDetected) {
    std::string output = readNVBitOutput();
    
    std::regex arg1Regex(R"(\s+\[output\] Arg 1 \(float\*\): (0x[0-9a-f]+), \[0\] = ([0-9.-]+))");
    std::smatch arg1Match;
    ASSERT_TRUE(std::regex_search(output, arg1Match, arg1Regex))
        << "Arg 1 output detection failed";
    
    EXPECT_TRUE(output.find("[output] Arg 1") != std::string::npos);
    
    std::regex outputPtrRegex(R"(\s+Output pointer: (0x[0-9a-f]+))");
    std::smatch outputPtrMatch;
    ASSERT_TRUE(std::regex_search(output, outputPtrMatch, outputPtrRegex))
        << "Output pointer not reported";
    
    EXPECT_EQ(arg1Match[1].str(), outputPtrMatch[1].str())
        << "Output pointer doesn't match Arg 1 address";
}

TEST_F(ArgumentDetectionTest, KernelLaunchDetected) {
    std::string output = readNVBitOutput();
    
    EXPECT_TRUE(output.find("Intercepted kernel launch: readWriteKernel(float*, float*)") != std::string::npos)
        << "Kernel launch not detected";
    
    std::regex gridRegex(R"(gridDim = \((\d+), (\d+), (\d+)\))");
    std::smatch gridMatch;
    ASSERT_TRUE(std::regex_search(output, gridMatch, gridRegex));
    
    EXPECT_EQ(std::stoi(gridMatch[1]), 4); 
    EXPECT_EQ(std::stoi(gridMatch[2]), 1);
    EXPECT_EQ(std::stoi(gridMatch[3]), 1);
    
    std::regex blockRegex(R"(blockDim = \((\d+), (\d+), (\d+)\))");
    std::smatch blockMatch;
    ASSERT_TRUE(std::regex_search(output, blockMatch, blockRegex));
    
    EXPECT_EQ(std::stoi(blockMatch[1]), 256); // blockDim.x
    EXPECT_EQ(std::stoi(blockMatch[2]), 1);   // blockDim.y
    EXPECT_EQ(std::stoi(blockMatch[3]), 1);   // blockDim.z
}

TEST_F(ArgumentDetectionTest, ArgumentTypesCorrect) {
    std::string output = readNVBitOutput();
    
    // Count argument types
    size_t inputCount = 0;
    size_t outputCount = 0;
    size_t scalarCount = 0;
    
    std::regex inputRegex(R"(\[input\] Arg)");
    std::regex outputRegex(R"(\[output\] Arg)");
    std::regex scalarRegex(R"(\[scalar\] Arg)");
    
    std::sregex_iterator inputIter(output.begin(), output.end(), inputRegex);
    std::sregex_iterator outputIter(output.begin(), output.end(), outputRegex);
    std::sregex_iterator scalarIter(output.begin(), output.end(), scalarRegex);
    std::sregex_iterator end;
    
    inputCount = std::distance(inputIter, end);
    outputCount = std::distance(outputIter, end);
    scalarCount = std::distance(scalarIter, end);
    
    // Verify expected counts for readWriteKernel(float* x, float* y)
    EXPECT_EQ(inputCount, 1) << "Should have 1 input argument";
    EXPECT_EQ(outputCount, 1) << "Should have 1 output argument";  
    EXPECT_EQ(scalarCount, 0) << "Should have 0 scalar arguments";
}

TEST_F(ArgumentDetectionTest, ArgumentDataTypesCorrect) {
    std::string output = readNVBitOutput();
    
    // Verify both arguments are float*
    EXPECT_TRUE(output.find("Arg 0 (float*)") != std::string::npos);
    EXPECT_TRUE(output.find("Arg 1 (float*)") != std::string::npos);
}

TEST_F(ArgumentDetectionTest, NoScalarArguments) {
    std::string output = readNVBitOutput();
    
    // Verify no scalar arguments for this kernel
    EXPECT_TRUE(output.find("[scalar]") == std::string::npos) 
        << "Should have no scalar arguments";
}

TEST_F(ArgumentDetectionTest, SuccessMessagePresent) {
    std::string output = readNVBitOutput();
    
    // Verify the program executed successfully
    EXPECT_TRUE(output.find("Success: all values match!") != std::string::npos)
        << "Program should report successful execution";
}
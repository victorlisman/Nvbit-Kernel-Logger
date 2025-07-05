#include <gtest/gtest.h>
#include <fstream>
#include <regex>
#include <string>

class ArgumentDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        system("LD_PRELOAD=../../tools/log_kernel_launch/log_kernel_launch.so ./vectoradd.elf > nvbit_output.txt 2>&1");
    }
    
    std::string readNVBitOutput() {
        std::ifstream file("nvbit_output.txt");
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        return content;
    }
};

TEST_F(ArgumentDetectionTest, InputArgumentsDetected) {
    std::string output = readNVBitOutput();
    
    std::regex arg0Regex(R"(\s+\[input\] Arg 0 \(float\*\): (0x[0-9a-f]+), \[0\] = ([0-9.-]+))");
    std::smatch arg0Match;
    ASSERT_TRUE(std::regex_search(output, arg0Match, arg0Regex)) 
        << "Arg 0 input detection failed";
    
    std::regex arg1Regex(R"(\s+\[input\] Arg 1 \(float\*\): (0x[0-9a-f]+), \[0\] = ([0-9.-]+))");
    std::smatch arg1Match;
    ASSERT_TRUE(std::regex_search(output, arg1Match, arg1Regex))
        << "Arg 1 input detection failed";
    
    EXPECT_TRUE(output.find("[input] Arg 0") != std::string::npos);
    EXPECT_TRUE(output.find("[input] Arg 1") != std::string::npos);
}

TEST_F(ArgumentDetectionTest, OutputArgumentDetected) {
    std::string output = readNVBitOutput();
    
    std::regex arg2Regex(R"(\s+\[output\] Arg 2 \(float\*\): (0x[0-9a-f]+), \[0\] = ([0-9.-]+))");
    std::smatch arg2Match;
    ASSERT_TRUE(std::regex_search(output, arg2Match, arg2Regex))
        << "Arg 2 output detection failed";
    
    EXPECT_TRUE(output.find("[output] Arg 2") != std::string::npos);
    
    std::regex outputPtrRegex(R"(\s+Output pointer: (0x[0-9a-f]+))");
    std::smatch outputPtrMatch;
    ASSERT_TRUE(std::regex_search(output, outputPtrMatch, outputPtrRegex))
        << "Output pointer not reported";
    
    EXPECT_EQ(arg2Match[1].str(), outputPtrMatch[1].str())
        << "Output pointer doesn't match Arg 2 address";
}

TEST_F(ArgumentDetectionTest, ScalarArgumentDetected) {
    std::string output = readNVBitOutput();
    
    std::regex arg3Regex(R"(\s+\[scalar\] Arg 3 \(int\): (\d+))");
    std::smatch arg3Match;
    ASSERT_TRUE(std::regex_search(output, arg3Match, arg3Regex))
        << "Arg 3 scalar detection failed";
    
    EXPECT_TRUE(output.find("[scalar] Arg 3") != std::string::npos);
    
    EXPECT_EQ(std::stoi(arg3Match[1].str()), 1024)
        << "Scalar value should be 1024";
}

TEST_F(ArgumentDetectionTest, ArgumentTypesCorrect) {
    std::string output = readNVBitOutput();
    
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
    
    EXPECT_EQ(inputCount, 2) << "Should have 2 input arguments";
    EXPECT_EQ(outputCount, 1) << "Should have 1 output argument";  
    EXPECT_EQ(scalarCount, 1) << "Should have 1 scalar argument";
}

TEST_F(ArgumentDetectionTest, ArgumentDataTypesCorrect) {
    std::string output = readNVBitOutput();
    
    EXPECT_TRUE(output.find("Arg 0 (float*)") != std::string::npos);
    EXPECT_TRUE(output.find("Arg 1 (float*)") != std::string::npos);
    EXPECT_TRUE(output.find("Arg 2 (float*)") != std::string::npos);
    
    EXPECT_TRUE(output.find("Arg 3 (int)") != std::string::npos);
}
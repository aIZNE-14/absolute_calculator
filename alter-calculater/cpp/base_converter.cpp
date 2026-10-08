#include <cstdint>
#include <cctype>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr char digits[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/";

int digit_value(char character, int base) {
    std::string alphabet(digits);
    std::size_t index = alphabet.find(character);
    if (index == std::string::npos && base <= 36) {
        const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
        index = alphabet.find(upper);
    }
    if (index == std::string::npos || index >= static_cast<std::size_t>(base)) {
        throw std::invalid_argument("Digit is not valid for this base.");
    }
    return static_cast<int>(index);
}

std::uint64_t parse_integer(std::string text, int base, bool& negative) {
    negative = false;
    if (!text.empty() && text.front() == '-') {
        negative = text.front() == '-';
        text.erase(text.begin());
    } else if (!text.empty() && text.front() == '+' && base < 63) {
        text.erase(text.begin());
    }
    if (text.empty()) throw std::invalid_argument("Enter a number.");
    std::uint64_t value = 0;
    for (char character : text) {
        const auto digit = static_cast<std::uint64_t>(digit_value(character, base));
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / base) {
            throw std::overflow_error("Integer exceeds uint64 range.");
        }
        value = value * base + digit;
    }
    return value;
}

std::string format_integer(std::uint64_t value, int base) {
    if (value == 0) return "0";
    std::string output;
    while (value != 0) {
        output.push_back(digits[value % base]);
        value /= base;
    }
    return std::string(output.rbegin(), output.rend());
}

bool supported(int base) {
    return base >= 2 && base <= 64;
}
}  // namespace

int main() {
    int source_base = 0;
    int target_base = 0;
    std::string input;
    if (!(std::cin >> source_base >> target_base >> input)) {
        std::cerr << "Usage: enter source-base target-base integer\n";
        return 2;
    }
    if (!supported(source_base) || !supported(target_base)) {
        std::cerr << "Bases must be 2, 8, 10, 16, 32, or 60.\n";
        return 2;
    }
    try {
        bool negative = false;
        const auto value = parse_integer(input, source_base, negative);
        std::cout << (negative && value != 0 ? "-" : "") << format_integer(value, target_base) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

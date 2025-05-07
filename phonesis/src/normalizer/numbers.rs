//! Number conversion module for text normalization
//! 
//! This module provides functionality to convert numeric values into their word representations,
//! supporting cardinal numbers, ordinal numbers, decimals, currency, fractions, and ranges.


/// Configuration options for number conversion
#[derive(Debug, Clone)]
pub struct NumberConverterOptions {
    /// Whether to use "and" between hundreds and tens (e.g., "one hundred and twenty")
    pub use_and: bool,
    /// Whether to use hyphens in compound numbers (e.g., "twenty-one" vs "twenty one")
    pub use_hyphens: bool,
    /// How to handle decimals (point vs dot)
    pub decimal_separator: String,
    /// Currency symbol mapping
    pub currency_symbols: Vec<(String, String)>,
}

impl Default for NumberConverterOptions {
    fn default() -> Self {
        Self {
            use_and: true,
            use_hyphens: true,
            decimal_separator: "point".to_string(),
            currency_symbols: vec![
                ("$".to_string(), "dollar".to_string()),
                ("€".to_string(), "euro".to_string()),
                ("£".to_string(), "pound".to_string()),
                ("¥".to_string(), "yen".to_string()),
            ],
        }
    }
}

/// Converts numeric values to their word representations
#[derive(Debug)]
pub struct NumberConverter {
    options: NumberConverterOptions,
}

impl NumberConverter {
    /// Creates a new number converter with default options
    pub fn new() -> Self {
        Self {
            options: NumberConverterOptions::default(),
        }
    }
    
    /// Creates a new number converter with custom options
    pub fn with_options(options: NumberConverterOptions) -> Self {
        Self { options }
    }
    
    /// Converts a cardinal number to words
    pub fn convert_cardinal(&self, number: i64) -> String {
        if number == 0 {
            return "zero".to_string();
        }
        
        if number < 0 {
            return format!("negative {}", self.convert_cardinal(-number));
        }
        
        self.convert_positive_cardinal(number)
    }
    
    /// Converts a positive cardinal number to words
    fn convert_positive_cardinal(&self, number: i64) -> String {
        let mut parts = Vec::new();
        let mut remaining = number;
        
        // Billions
        if remaining >= 1_000_000_000 {
            let billions = remaining / 1_000_000_000;
            parts.push(format!("{} billion", self.convert_small_number(billions)));
            remaining %= 1_000_000_000;
        }
        
        // Millions
        if remaining >= 1_000_000 {
            let millions = remaining / 1_000_000;
            parts.push(format!("{} million", self.convert_small_number(millions)));
            remaining %= 1_000_000;
        }
        
        // Thousands
        if remaining >= 1_000 {
            let thousands = remaining / 1_000;
            parts.push(format!("{} thousand", self.convert_small_number(thousands)));
            remaining %= 1_000;
        }
        
        // Hundreds and remainder
        if remaining > 0 {
            parts.push(self.convert_small_number(remaining));
        }
        
        parts.join(" ")
    }
    
    /// Converts numbers less than 1000
    fn convert_small_number(&self, number: i64) -> String {
        if number >= 100 {
            let hundreds = number / 100;
            let remainder = number % 100;
            let mut result = format!("{} hundred", self.convert_units(hundreds));
            
            if remainder > 0 {
                if self.options.use_and {
                    result.push_str(" and ");
                } else {
                    result.push(' ');
                }
                result.push_str(&self.convert_two_digit_number(remainder));
            }
            
            result
        } else {
            self.convert_two_digit_number(number)
        }
    }
    
    /// Converts two-digit numbers
    fn convert_two_digit_number(&self, number: i64) -> String {
        if number < 20 {
            self.convert_units(number)
        } else {
            let tens = number / 10;
            let units = number % 10;
            let mut result = self.convert_tens(tens);
            
            if units > 0 {
                if self.options.use_hyphens {
                    result.push('-');
                } else {
                    result.push(' ');
                }
                result.push_str(&self.convert_units(units));
            }
            
            result
        }
    }
    
    /// Converts single digit numbers and teens
    fn convert_units(&self, number: i64) -> String {
        match number {
            0 => "zero",
            1 => "one",
            2 => "two",
            3 => "three",
            4 => "four",
            5 => "five",
            6 => "six",
            7 => "seven",
            8 => "eight",
            9 => "nine",
            10 => "ten",
            11 => "eleven",
            12 => "twelve",
            13 => "thirteen",
            14 => "fourteen",
            15 => "fifteen",
            16 => "sixteen",
            17 => "seventeen",
            18 => "eighteen",
            19 => "nineteen",
            _ => "unknown",
        }.to_string()
    }
    
    /// Converts tens place values
    fn convert_tens(&self, tens: i64) -> String {
        match tens {
            2 => "twenty",
            3 => "thirty",
            4 => "forty",
            5 => "fifty",
            6 => "sixty",
            7 => "seventy",
            8 => "eighty",
            9 => "ninety",
            _ => "unknown",
        }.to_string()
    }
    
    /// Converts an ordinal number to words
    pub fn convert_ordinal(&self, number: i64) -> String {
        if number == 0 {
            return "zeroth".to_string();
        }
        
        if number < 0 {
            return format!("negative {}", self.convert_ordinal(-number));
        }
        
        // Special cases for ordinals
        match number {
            1 => "first".to_string(),
            2 => "second".to_string(),
            3 => "third".to_string(),
            5 => "fifth".to_string(),
            8 => "eighth".to_string(),
            9 => "ninth".to_string(),
            12 => "twelfth".to_string(),
            _ => {
                let cardinal = self.convert_cardinal(number);
                if number % 100 >= 20 {
                    // Handle compound numbers like twenty-first
                    let last_digit = number % 10;
                    if last_digit == 0 {
                        format!("{}th", cardinal)
                    } else {
                        // For hyphenated numbers, replace the last word
                        if self.options.use_hyphens && cardinal.contains('-') {
                            let parts: Vec<_> = cardinal.rsplitn(2, '-').collect();
                            if parts.len() == 2 {
                                let ordinal_suffix = match parts[0] {
                                    "one" => "first",
                                    "two" => "second",
                                    "three" => "third",
                                    "five" => "fifth",
                                    "eight" => "eighth",
                                    "nine" => "ninth",
                                    _ => {
                                        return format!("{}th", cardinal);
                                    }
                                };
                                return format!("{}-{}", parts[1], ordinal_suffix);
                            }
                        }
                        
                        // For non-hyphenated or simple cases
                        let mut parts: Vec<&str> = cardinal.split_whitespace().collect();
                        if let Some(last) = parts.last_mut() {
                            *last = match *last {
                                "one" => "first",
                                "two" => "second",
                                "three" => "third",
                                "five" => "fifth",
                                "eight" => "eighth",
                                "nine" => "ninth",
                                _ => return format!("{}th", cardinal),
                            };
                        }
                        parts.join(" ")
                    }
                } else {
                    format!("{}th", cardinal)
                }
            }
        }
    }
    
    /// Converts a decimal number to words
    pub fn convert_decimal(&self, number: f64) -> String {
        let number_str = number.to_string();
        let parts: Vec<&str> = number_str.split('.').collect();
        
        let integer_part = match parts[0].parse::<i64>() {
            Ok(num) => self.convert_cardinal(num),
            Err(_) => parts[0].to_string(),
        };
        
        if parts.len() == 1 {
            // No decimal places
            integer_part
        } else {
            // Extract up to two decimal places for standard representation
            let decimal_str = parts[1];
            let decimal_digits = if decimal_str.len() >= 2 {
                &decimal_str[..2]
            } else {
                decimal_str
            };
            
            // Parse as an integer to get the numerical value
            let decimal_value = decimal_digits.parse::<i64>().unwrap_or(0);
            
            // If decimal value is 0, just return the integer part
            if decimal_value == 0 {
                return integer_part;
            }
            
            // Convert the fractional part
            let fractional_part = if decimal_digits.len() == 1 {
                // Single digit, e.g., 0.5 becomes "fifty" (treat as 50)
                self.convert_cardinal(decimal_value * 10)
            } else {
                // Two or more digits, e.g., 0.14 becomes "fourteen"
                if decimal_digits.len() >= 2 && decimal_digits.starts_with("0") && !decimal_digits.ends_with("0") {
                    // Special case for numbers like "05" which should be "zero five"
                    let first_digit = decimal_digits.chars().nth(0).unwrap().to_digit(10).unwrap() as i64;
                    let second_digit = decimal_digits.chars().nth(1).unwrap().to_digit(10).unwrap() as i64;
                    
                    if first_digit == 0 {
                        format!("zero {}", self.convert_units(second_digit))
                    } else {
                        self.convert_cardinal(decimal_value)
                    }
                } else {
                    self.convert_cardinal(decimal_value)
                }
            };
            
            format!(
                "{} {} {}",
                integer_part,
                self.options.decimal_separator,
                fractional_part
            )
        }
    }
    
    /// Converts a currency value to words
    pub fn convert_currency(&self, amount: f64, symbol: &str) -> String {
        let currency_name = self.options.currency_symbols.iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, name)| name.as_str())
            .unwrap_or("currency");
        
        let integer_part = amount.trunc() as i64;
        let cents = (amount.fract() * 100.0).round() as i64;
        
        let mut result = format!(
            "{} {}{}",
            self.convert_cardinal(integer_part),
            currency_name,
            if integer_part == 1 { "" } else { "s" }
        );
        
        if cents > 0 {
            result.push_str(&format!(
                " and {} cent{}",
                self.convert_cardinal(cents),
                if cents == 1 { "" } else { "s" }
            ));
        }
        
        result
    }
    
    /// Converts a fraction to words
    pub fn convert_fraction(&self, numerator: i64, denominator: i64) -> String {
        if denominator == 0 {
            return "undefined".to_string();
        }
        
        if numerator == 0 {
            return "zero".to_string();
        }
        
        if denominator == 1 {
            return self.convert_cardinal(numerator);
        }
        
        let numerator_word = self.convert_cardinal(numerator);
        let denominator_word = match denominator {
            2 => "half".to_string(),
            3 => "third".to_string(),
            4 => "quarter".to_string(),
            _ => self.convert_ordinal(denominator),
        };
        
        if numerator == 1 {
            format!("one {}", denominator_word)
        } else {
            format!("{} {}s", numerator_word, denominator_word)
        }
    }
    
    /// Converts a range expression to words
    pub fn convert_range(&self, start: i64, end: i64) -> String {
        format!(
            "{} to {}",
            self.convert_cardinal(start),
            self.convert_cardinal(end)
        )
    }
    
    /// Converts a basic math expression to words
    pub fn convert_math(&self, num1: i64, operator: &str, num2: i64) -> String {
        let operator_word = match operator {
            "+" => "plus",
            "-" => "minus",
            "*" | "×" => "times",
            "/" | "÷" => "divided by",
            _ => operator,
        };
        
        format!(
            "{} {} {}",
            self.convert_cardinal(num1),
            operator_word,
            self.convert_cardinal(num2)
        )
    }
}

impl Default for NumberConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cardinal_conversion() {
        let converter = NumberConverter::new();
        
        assert_eq!(converter.convert_cardinal(0), "zero");
        assert_eq!(converter.convert_cardinal(1), "one");
        assert_eq!(converter.convert_cardinal(11), "eleven");
        assert_eq!(converter.convert_cardinal(21), "twenty-one");
        assert_eq!(converter.convert_cardinal(42), "forty-two");
        assert_eq!(converter.convert_cardinal(100), "one hundred");
        assert_eq!(converter.convert_cardinal(123), "one hundred and twenty-three");
        assert_eq!(converter.convert_cardinal(1000), "one thousand");
        assert_eq!(converter.convert_cardinal(1234), "one thousand two hundred and thirty-four");
        assert_eq!(converter.convert_cardinal(1_000_000), "one million");
        assert_eq!(converter.convert_cardinal(1_000_000_000), "one billion");
    }
    
    #[test]
    fn test_negative_numbers() {
        let converter = NumberConverter::new();
        
        assert_eq!(converter.convert_cardinal(-1), "negative one");
        assert_eq!(converter.convert_cardinal(-42), "negative forty-two");
    }
    
    #[test]
    fn test_ordinal_conversion() {
        let converter = NumberConverter::new();
        
        assert_eq!(converter.convert_ordinal(1), "first");
        assert_eq!(converter.convert_ordinal(2), "second");
        assert_eq!(converter.convert_ordinal(3), "third");
        assert_eq!(converter.convert_ordinal(4), "fourth");
        assert_eq!(converter.convert_ordinal(11), "eleventh");
        assert_eq!(converter.convert_ordinal(21), "twenty-first");
        assert_eq!(converter.convert_ordinal(42), "forty-second");
        assert_eq!(converter.convert_ordinal(100), "one hundredth");
    }
    
    #[test]
    fn test_decimal_conversion() {
        let converter = NumberConverter::new();
        
        assert_eq!(converter.convert_decimal(3.14), "three point fourteen");
        assert_eq!(converter.convert_decimal(0.5), "zero point fifty");
        assert_eq!(converter.convert_decimal(1.0), "one");
        assert_eq!(converter.convert_decimal(10.05), "ten point zero five");
    }
    
    #[test]
    fn test_currency_conversion() {
        let converter = NumberConverter::new();
        
        assert_eq!(converter.convert_currency(10.00, "$"), "ten dollars");
        assert_eq!(converter.convert_currency(1.00, "$"), "one dollar");
        assert_eq!(converter.convert_currency(10.99, "$"), "ten dollars and ninety-nine cents");
        assert_eq!(converter.convert_currency(0.01, "$"), "zero dollars and one cent");
        assert_eq!(converter.convert_currency(5.50, "€"), "five euros and fifty cents");
    }
    
    #[test]
    fn test_fraction_conversion() {
        let converter = NumberConverter::new();
        
        assert_eq!(converter.convert_fraction(1, 2), "one half");
        assert_eq!(converter.convert_fraction(3, 4), "three quarters");
        assert_eq!(converter.convert_fraction(2, 3), "two thirds");
        assert_eq!(converter.convert_fraction(5, 8), "five eighths");
        assert_eq!(converter.convert_fraction(0, 5), "zero");
        assert_eq!(converter.convert_fraction(5, 0), "undefined");
    }
    
    #[test]
    fn test_range_conversion() {
        let converter = NumberConverter::new();
        
        assert_eq!(converter.convert_range(1, 5), "one to five");
        assert_eq!(converter.convert_range(10, 20), "ten to twenty");
    }
    
    #[test]
    fn test_math_conversion() {
        let converter = NumberConverter::new();
        
        assert_eq!(converter.convert_math(2, "+", 3), "two plus three");
        assert_eq!(converter.convert_math(5, "-", 2), "five minus two");
        assert_eq!(converter.convert_math(4, "*", 6), "four times six");
        assert_eq!(converter.convert_math(10, "/", 2), "ten divided by two");
    }
    
    #[test]
    fn test_custom_options() {
        let options = NumberConverterOptions {
            use_and: false,
            use_hyphens: false,
            decimal_separator: "dot".to_string(),
            ..Default::default()
        };
        let converter = NumberConverter::with_options(options);
        
        assert_eq!(converter.convert_cardinal(123), "one hundred twenty three");
        assert_eq!(converter.convert_cardinal(21), "twenty one");
        assert_eq!(converter.convert_decimal(3.14), "three dot fourteen");
    }
}
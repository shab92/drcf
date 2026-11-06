// Australian national public holidays 2011-2013 (python `holidays` 0.40, holidays.Australia, no state),
// the list Prophet's add_country_holidays(country_name='AU') uses. One indicator per holiday name.
#pragma once
#include <array>
#include <string>
#include <vector>
namespace au_holidays {
inline const std::vector<std::string> names = {"Anzac Day", "Australia Day", "Australia Day (Observed)", "Boxing Day", "Christmas Day", "Christmas Day (Observed)", "Easter Monday", "Good Friday", "New Year's Day", "New Year's Day (Observed)"};
struct Day { int y, m, d, name; };
inline const std::vector<Day> days = {
    {2011, 1, 1, 8},  // New Year's Day
    {2011, 1, 3, 9},  // New Year's Day (Observed)
    {2011, 1, 26, 1},  // Australia Day
    {2011, 4, 22, 7},  // Good Friday
    {2011, 4, 25, 0},  // Anzac Day
    {2011, 4, 25, 6},  // Easter Monday
    {2011, 12, 25, 4},  // Christmas Day
    {2011, 12, 26, 3},  // Boxing Day
    {2011, 12, 27, 5},  // Christmas Day (Observed)
    {2012, 1, 1, 8},  // New Year's Day
    {2012, 1, 2, 9},  // New Year's Day (Observed)
    {2012, 1, 26, 1},  // Australia Day
    {2012, 4, 6, 7},  // Good Friday
    {2012, 4, 9, 6},  // Easter Monday
    {2012, 4, 25, 0},  // Anzac Day
    {2012, 12, 25, 4},  // Christmas Day
    {2012, 12, 26, 3},  // Boxing Day
    {2013, 1, 1, 8},  // New Year's Day
    {2013, 1, 26, 1},  // Australia Day
    {2013, 1, 28, 2},  // Australia Day (Observed)
    {2013, 3, 29, 7},  // Good Friday
    {2013, 4, 1, 6},  // Easter Monday
    {2013, 4, 25, 0},  // Anzac Day
    {2013, 12, 25, 4},  // Christmas Day
    {2013, 12, 26, 3},  // Boxing Day
};
}  // namespace au_holidays

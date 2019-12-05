#include <regex>
#include <fstream>
#include <cassert>
#include <iostream>
#include <chrono>

void Init(const std::string & config_file)
{
  std::ifstream in(config_file);
  std::string line;
  std::regex e1 ("tolerance");
  std::regex e2 (".*\\s*=\\s*(.+)");
  std::smatch m;
  while(std::getline(in, line))
    {
      if(std::regex_search(line, m, e1))
        {
          double tolerance = std::stod(std::regex_replace(line, e2, "$1"));
	  std::cout << tolerance << std::endl;
        }
    }
  in.close();
}

int main()
{
  Init("test.in");
}

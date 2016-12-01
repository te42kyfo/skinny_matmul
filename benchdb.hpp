#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include "sqlite3.h"

using StrPair = std::pair<std::string, std::string>;
using StrPairList = std::vector<StrPair>;

struct Entry {
  StrPairList keys;
  StrPairList values;
};

class BenchDB {
 public:
  BenchDB(std::string dbFileName) : _dbFileName(dbFileName) {
    //    std::cout << "BenchDB: open db " << dbFileName << "\n";
    int result = sqlite3_open(_dbFileName.c_str(), &_dbCon);

    if (!_dbCon || result != SQLITE_OK) {
      std::cerr << "BenchDB: sqlite3_open Error\n";
      const char* errMsg = sqlite3_errmsg(_dbCon);
      std::cerr << errMsg << "\n";
      sqlite3_close(_dbCon);
      _dbCon = NULL;
    }
    sqlite3_busy_timeout(_dbCon, 10000);
  }
  ~BenchDB() {
    flush_entries();
    // std::cout << "BenchDB: close database\n";
    int result = sqlite3_close(_dbCon);
    if (result != SQLITE_OK) {
      std::cerr << "BenchDB: sqlite3_close Error\n";
      const char* errMsg = sqlite3_errmsg(_dbCon);
      std::cerr << errMsg << "\n";
    }
  }
  BenchDB(const BenchDB&) = delete;

  void insert(StrPairList keys, StrPairList values) {
    _queue.push_back({keys, values});
    if (_queue.size() > 50) flush_entries();
  }

 private:
  void flush_entries() {
    sqlite3_exec(_dbCon, "BEGIN TRANSACTION", NULL, NULL, NULL);
    for (const auto& e : _queue) insert_entry(e);
    _queue.clear();
    sqlite3_exec(_dbCon, "COMMIT TRANSACTION", NULL, NULL, NULL);
  }
  void insert_entry(const Entry& e) {
    if (_dbCon == NULL) return;
    std::stringstream query;

    query << "DELETE FROM benchmarks WHERE ";
    for (const auto& key : e.keys) {
      query << key.first << "=" << key.second << " AND ";
    }
    query << " 1=1";

    std::string queryStr = query.str();
    sqlite3_error(sqlite3_exec(_dbCon, queryStr.c_str(), NULL, NULL, NULL));

    query.str("");
    query << "INSERT INTO benchmarks (";
    for (const auto& key : e.keys) {
      query << key.first << ", ";
    }
    for (const auto& value : e.values) {
      query << value.first << ", ";
    }
    query.seekp(-2, std::ios_base::end);
    query << ") VALUES (";
    for (const auto& key : e.keys) {
      query << key.second << ", ";
    }
    for (const auto& value : e.values) {
      query << value.second << ", ";
    }
    query.seekp(-2, std::ios_base::end);
    query << ")";
    queryStr = query.str();
    sqlite3_error(sqlite3_exec(_dbCon, queryStr.c_str(), NULL, NULL, NULL));
  }

  void sqlite3_error(int result) {
    if (result != SQLITE_OK) {
      std::cerr << "BenchDB: sqlite Error\n";
      const char* errMsg = sqlite3_errmsg(_dbCon);
      std::cerr << errMsg << "\n";
    }
  }

  std::vector<Entry> _queue;
  sqlite3* _dbCon;
  std::string _dbFileName;
};

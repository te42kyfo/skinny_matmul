#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include "sqlite3.h"

struct Entry {
  int M, N;
  std::string name, mode;
  bool inplace, zerobeta;
  int K;
  double time, flops, bw;
};

class BenchDB {
 public:
  BenchDB(std::string dbFileName) : _dbFileName(dbFileName) {
    std::cout << "BenchDB: open db " << dbFileName << "\n";
    int result = sqlite3_open(_dbFileName.c_str(), &_dbCon);

    if (!_dbCon || result != SQLITE_OK) {
      std::cerr << "BenchDB: sqlite3_open Error\n";
      const char* errMsg = sqlite3_errmsg(_dbCon);
      std::cerr << errMsg << "\n";
      sqlite3_close(_dbCon);
      _dbCon = NULL;
    }
  }
  ~BenchDB() {
    flush_entries();
    std::cout << "BenchDB: close database\n";
    int result = sqlite3_close(_dbCon);
    if (result != SQLITE_OK) {
      std::cerr << "BenchDB: sqlite3_close Error\n";
      const char* errMsg = sqlite3_errmsg(_dbCon);
      std::cerr << errMsg << "\n";
    }
  }
  BenchDB(const BenchDB&) = delete;

  void insert(int M, int N, std::string name, std::string mode, bool inplace,
              bool zerobeta, int K, double time, double flops, double bw) {
    _queue.push_back({M, N, name, mode, inplace, zerobeta, K, time, flops, bw});
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

    query << "INSERT OR REPLACE INTO tsmm (M, N, name, inplace, zerobeta, K, "
             "time, "
             "flops, "
             "bw) values("
          << e.M << ", " << e.N << ", \"" << e.name << "\", " << e.inplace
          << ", " << e.zerobeta << ", " << e.K << ", " << e.time << ", "
          << e.flops << ", " << e.bw << ")";

    std::string queryStr = query.str();
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

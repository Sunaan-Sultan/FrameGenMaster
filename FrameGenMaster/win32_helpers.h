#pragma once
#include <windows.h>
#include <vector>
#include <string>

struct WinInfo {
    HWND hwnd;
    char title[256];
};

void RefreshWindowList(std::vector<WinInfo>& outList);
bool CaptureWindowToBitmap(HWND target, void* mappedData,
    int& outW, int& outH);

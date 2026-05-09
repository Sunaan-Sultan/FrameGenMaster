#include "win32_helpers.h"
#include <tlhelp32.h>
#include <cstring>

static std::vector<WinInfo>* s_list = nullptr;

static BOOL CALLBACK EnumProc(HWND hwnd, LPARAM) {
    if (!IsWindowVisible(hwnd)) return TRUE;
    char title[256] = {};
    GetWindowTextA(hwnd, title, sizeof(title));
    if (strlen(title) < 2) return TRUE;
    if (hwnd == GetConsoleWindow()) return TRUE;
    WinInfo w; w.hwnd = hwnd;
    strcpy_s(w.title, title);
    s_list->push_back(w);
    return TRUE;
}

void RefreshWindowList(std::vector<WinInfo>& outList) {
    outList.clear();
    s_list = &outList;
    EnumWindows(EnumProc, 0);
    s_list = nullptr;
}

bool CaptureWindowToBitmap(HWND target, void* mappedData,
    int& outW, int& outH)
{
    RECT r{};
    if (!GetClientRect(target, &r)) return false;
    int W = r.right - r.left;
    int H = r.bottom - r.top;
    if (W <= 0 || H <= 0) return false;
    outW = W; outH = H;

    HDC hdcWin = GetDC(target);
    HDC hdcMem = CreateCompatibleDC(hdcWin);
    HBITMAP hbm = CreateCompatibleBitmap(hdcWin, W, H);
    SelectObject(hdcMem, hbm);
    PrintWindow(target, hdcMem, PW_CLIENTONLY);

    BITMAPINFOHEADER bi{};
    bi.biSize = sizeof(bi);
    bi.biWidth = W;
    bi.biHeight = -H;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;

    GetDIBits(hdcMem, hbm, 0, H,
        mappedData, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    DeleteObject(hbm);
    DeleteDC(hdcMem);
    ReleaseDC(target, hdcWin);
    return true;
}
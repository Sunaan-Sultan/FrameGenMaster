#include <vector>
#include <string>
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_d3d11_interop.h>

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <tchar.h>

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "win32_helpers.h"

#pragma comment(lib, "d3d11.lib")

// ==========================================
// LOGGER
// ==========================================
std::vector<std::string> g_Logs;
void AppLog(const std::string& msg) {
    if (g_Logs.size() > 30) g_Logs.erase(g_Logs.begin());
    g_Logs.push_back(msg);
}

// ==========================================
// CUDA KERNELS
// Desktop Duplication দেয় BGRA — kernel এ
// সরাসরি BGRA → float plane, কোনো swap নেই
// ==========================================
__global__ void bgraToFloatKernel(
    cudaTextureObject_t tex,
    float* outB, float* outG, float* outR,
    int srcX, int srcY,   // crop offset in full desktop tex
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uchar4 p = tex2D<uchar4>(tex, x + srcX, y + srcY);
    int idx = y * W + x;
    outB[idx] = p.x / 255.f; // B
    outG[idx] = p.y / 255.f; // G
    outR[idx] = p.z / 255.f; // R
}

__global__ void blendKernel(
    const float* aB, const float* aG, const float* aR,
    const float* bB, const float* bG, const float* bR,
    float* outB, float* outG, float* outR,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    outB[idx] = (aB[idx] + bB[idx]) * 0.5f;
    outG[idx] = (aG[idx] + bG[idx]) * 0.5f;
    outR[idx] = (aR[idx] + bR[idx]) * 0.5f;
}

__global__ void floatToBgraKernel(
    const float* inB, const float* inG, const float* inR,
    cudaSurfaceObject_t surf,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    uchar4 p;
    p.x = (unsigned char)(fminf(fmaxf(inB[idx], 0.f), 1.f) * 255.f); // B
    p.y = (unsigned char)(fminf(fmaxf(inG[idx], 0.f), 1.f) * 255.f); // G
    p.z = (unsigned char)(fminf(fmaxf(inR[idx], 0.f), 1.f) * 255.f); // R
    p.w = 255;
    surf2Dwrite(p, surf, x * sizeof(uchar4), y);
}

// ==========================================
// D3D11 GLOBALS (single device)
// ==========================================
static ID3D11Device* g_Dev = nullptr;
static ID3D11DeviceContext* g_Ctx = nullptr;
static IDXGISwapChain* g_UiSC = nullptr;
static ID3D11RenderTargetView* g_UiRTV = nullptr;
static IDXGISwapChain* g_OvSC = nullptr;
static ID3D11RenderTargetView* g_OvRTV = nullptr;
static HWND g_hwndUI = nullptr;
static HWND g_hwndOv = nullptr;

// DXGI Desktop Duplication
static IDXGIOutputDuplication* g_DeskDupl = nullptr;

// Frame gen — 3 separate planes per frame (B/G/R)
static float* g_aB = nullptr; static float* g_aG = nullptr; static float* g_aR = nullptr;
static float* g_bB = nullptr; static float* g_bG = nullptr; static float* g_bR = nullptr;
static float* g_oB = nullptr; static float* g_oG = nullptr; static float* g_oR = nullptr;
static int    g_fgW = 0, g_fgH = 0;
static bool   g_firstFrame = true;

// Output textures for ImGui display
static ID3D11Texture2D* g_OrigTex = nullptr;
static ID3D11ShaderResourceView* g_OrigSRV = nullptr;
static ID3D11Texture2D* g_DisplayTex = nullptr;
static ID3D11ShaderResourceView* g_DisplaySRV = nullptr;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProcUI(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;
    if (msg == WM_SIZE && g_Dev && wParam != SIZE_MINIMIZED) {
        if (g_UiRTV) { g_UiRTV->Release(); g_UiRTV = nullptr; }
        g_UiSC->ResizeBuffers(0, LOWORD(lParam), HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
        ID3D11Texture2D* bb; g_UiSC->GetBuffer(0, IID_PPV_ARGS(&bb));
        g_Dev->CreateRenderTargetView(bb, nullptr, &g_UiRTV); bb->Release(); return 0;
    }
    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}
LRESULT WINAPI WndProcOv(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(h, m, w, l);
}

// ==========================================
// FREE ALL FRAME GEN RESOURCES
// ==========================================
static void FreeFrameGen() {
    auto cf = [](float*& p) {if (p) { cudaFree(p); p = nullptr; }};
    cf(g_aB); cf(g_aG); cf(g_aR);
    cf(g_bB); cf(g_bG); cf(g_bR);
    cf(g_oB); cf(g_oG); cf(g_oR);
    if (g_OrigSRV) { g_OrigSRV->Release();   g_OrigSRV = nullptr; }
    if (g_OrigTex) { g_OrigTex->Release();   g_OrigTex = nullptr; }
    if (g_DisplaySRV) { g_DisplaySRV->Release(); g_DisplaySRV = nullptr; }
    if (g_DisplayTex) { g_DisplayTex->Release(); g_DisplayTex = nullptr; }
    g_fgW = g_fgH = 0; g_firstFrame = true;
}

// ==========================================
// ENSURE BUFFERS FOR GIVEN SIZE
// ==========================================
static void EnsureBuffers(int W, int H, DXGI_FORMAT fmt) {
    if (W == g_fgW && H == g_fgH) return;
    FreeFrameGen();

    size_t planeBytes = (size_t)W * H * sizeof(float);
    cudaMalloc(&g_aB, planeBytes); cudaMalloc(&g_aG, planeBytes); cudaMalloc(&g_aR, planeBytes);
    cudaMalloc(&g_bB, planeBytes); cudaMalloc(&g_bG, planeBytes); cudaMalloc(&g_bR, planeBytes);
    cudaMalloc(&g_oB, planeBytes); cudaMalloc(&g_oG, planeBytes); cudaMalloc(&g_oR, planeBytes);

    // Use EXACT format from desktop texture — no mismatch possible
    D3D11_TEXTURE2D_DESC td{};
    td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = fmt; // exact desktop format
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;

    g_Dev->CreateTexture2D(&td, nullptr, &g_OrigTex);
    g_Dev->CreateShaderResourceView(g_OrigTex, nullptr, &g_OrigSRV);
    g_Dev->CreateTexture2D(&td, nullptr, &g_DisplayTex);
    g_Dev->CreateShaderResourceView(g_DisplayTex, nullptr, &g_DisplaySRV);

    g_fgW = W; g_fgH = H; g_firstFrame = true;
    AppLog("[OK] " + std::to_string(W) + "x" + std::to_string(H) + " buffers ready.");
}

// ==========================================
// SETUP DXGI DUPLICATION
// ==========================================
static bool SetupDuplication() {
    if (g_DeskDupl) { g_DeskDupl->Release(); g_DeskDupl = nullptr; }
    IDXGIDevice* dd = nullptr; IDXGIAdapter* da = nullptr;
    IDXGIOutput* dout = nullptr; IDXGIOutput1* dout1 = nullptr;
    g_Dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dd);
    dd->GetParent(__uuidof(IDXGIAdapter), (void**)&da);
    da->EnumOutputs(0, &dout);
    dout->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dout1);
    HRESULT hr = dout1->DuplicateOutput(g_Dev, &g_DeskDupl);
    dout1->Release(); dout->Release(); da->Release(); dd->Release();
    if (FAILED(hr)) { AppLog("[ERR] DuplicateOutput failed."); return false; }
    AppLog("[OK] Desktop Duplication ready.");
    return true;
}

// ==========================================
// CAPTURE + PROCESS
// 0 = no frame, 1 = orig only, 2 = orig+interp
// ==========================================
static int CaptureAndProcess(HWND targetWnd)
{
    if (!g_DeskDupl) return 0;

    RECT wr{};
    if (!GetWindowRect(targetWnd, &wr)) return 0;
    int sW = GetSystemMetrics(SM_CXSCREEN);
    int sH = GetSystemMetrics(SM_CYSCREEN);
    wr.left = max(wr.left, 0L); wr.top = max(wr.top, 0L);
    wr.right = min(wr.right, (LONG)sW);
    wr.bottom = min(wr.bottom, (LONG)sH);
    int cropX = wr.left, cropY = wr.top;
    int cropW = wr.right - wr.left, cropH = wr.bottom - wr.top;
    if (cropW <= 0 || cropH <= 0) return 0;

    // Try acquire new frame
    IDXGIResource* res = nullptr;
    DXGI_OUTDUPL_FRAME_INFO fi{};
    HRESULT hr = g_DeskDupl->AcquireNextFrame(16, &fi, &res);

    if (hr == DXGI_ERROR_ACCESS_LOST) {
        AppLog("[Warn] Access lost, reconnecting...");
        SetupDuplication(); return 0;
    }

    bool hasNewFrame = SUCCEEDED(hr) && fi.LastPresentTime.QuadPart != 0;

    if (hasNewFrame) {
        ID3D11Texture2D* deskTex = nullptr;
        res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&deskTex);

        // Get EXACT format from desktop texture
        D3D11_TEXTURE2D_DESC deskDesc{};
        deskTex->GetDesc(&deskDesc);

        // Ensure our buffers match desktop format exactly
        EnsureBuffers(cropW, cropH, deskDesc.Format);

        // Copy cropped region from desktop → g_OrigTex
        D3D11_BOX box{};
        box.left = (UINT)cropX; box.top = (UINT)cropY; box.front = 0;
        box.right = (UINT)wr.right; box.bottom = (UINT)wr.bottom; box.back = 1;
        g_Ctx->CopySubresourceRegion(g_OrigTex, 0, 0, 0, 0, deskTex, 0, &box);

        deskTex->Release();
        res->Release();
        g_DeskDupl->ReleaseFrame();

        // g_OrigTex → CUDA → float planes
        cudaGraphicsResource* capRes = nullptr;
        if (cudaGraphicsD3D11RegisterResource(&capRes, g_OrigTex,
            cudaGraphicsRegisterFlagsNone) != cudaSuccess) return 1;

        cudaGraphicsMapResources(1, &capRes, 0);
        cudaArray_t capArr = nullptr;
        cudaGraphicsSubResourceGetMappedArray(&capArr, capRes, 0, 0);

        cudaResourceDesc rd{}; rd.resType = cudaResourceTypeArray;
        rd.res.array.array = capArr;
        cudaTextureDesc td{};
        td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode = cudaFilterModePoint;
        td.readMode = cudaReadModeElementType;
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &rd, &td, nullptr);

        dim3 block(16, 16), grid((cropW + 15) / 16, (cropH + 15) / 16);

        // Write into frame B buffers (current frame)
        float* dstB = g_firstFrame ? g_aB : g_bB;
        float* dstG = g_firstFrame ? g_aG : g_bG;
        float* dstR = g_firstFrame ? g_aR : g_bR;
        // crop already done via CopySubresourceRegion, so srcX=srcY=0
        bgraToFloatKernel << <grid, block >> > (texObj, dstB, dstG, dstR, 0, 0, cropW, cropH);
        cudaDeviceSynchronize();

        cudaDestroyTextureObject(texObj);
        cudaGraphicsUnmapResources(1, &capRes, 0);
        cudaGraphicsUnregisterResource(capRes);

        if (!g_firstFrame) {
            // Blend A+B → output
            blendKernel << <grid, block >> > (
                g_aB, g_aG, g_aR,
                g_bB, g_bG, g_bR,
                g_oB, g_oG, g_oR,
                cropW, cropH);
            cudaDeviceSynchronize();

            // output float → g_DisplayTex
            cudaGraphicsResource* dr = nullptr;
            cudaGraphicsD3D11RegisterResource(&dr, g_DisplayTex,
                cudaGraphicsRegisterFlagsSurfaceLoadStore);
            cudaGraphicsMapResources(1, &dr, 0);
            cudaArray_t da = nullptr;
            cudaGraphicsSubResourceGetMappedArray(&da, dr, 0, 0);
            cudaResourceDesc srd{}; srd.resType = cudaResourceTypeArray;
            srd.res.array.array = da;
            cudaSurfaceObject_t surf = 0;
            cudaCreateSurfaceObject(&surf, &srd);
            floatToBgraKernel << <grid, block >> > (g_oB, g_oG, g_oR, surf, cropW, cropH);
            cudaDeviceSynchronize();
            cudaDestroySurfaceObject(surf);
            cudaGraphicsUnmapResources(1, &dr, 0);
            cudaGraphicsUnregisterResource(dr);

            // Swap A↔B for next frame
            auto sw = [](float*& x, float*& y) {float* t = x; x = y; y = t; };
            sw(g_aB, g_bB); sw(g_aG, g_bG); sw(g_aR, g_bR);
            g_firstFrame = false;
            return 2;
        }

        auto sw = [](float*& x, float*& y) {float* t = x; x = y; y = t; };
        sw(g_aB, g_bB); sw(g_aG, g_bG); sw(g_aR, g_bR);
        g_firstFrame = false;
        return 1;

    }
    else {
        // Timeout — no new desktop frame
        if (res) { res->Release(); g_DeskDupl->ReleaseFrame(); }
        // Still generate interpolated from last two frames
        if (!g_firstFrame && g_fgW > 0) {
            dim3 block(16, 16), grid((g_fgW + 15) / 16, (g_fgH + 15) / 16);
            blendKernel << <grid, block >> > (
                g_aB, g_aG, g_aR,
                g_bB, g_bG, g_bR,
                g_oB, g_oG, g_oR,
                g_fgW, g_fgH);
            cudaDeviceSynchronize();

            cudaGraphicsResource* dr = nullptr;
            cudaGraphicsD3D11RegisterResource(&dr, g_DisplayTex,
                cudaGraphicsRegisterFlagsSurfaceLoadStore);
            cudaGraphicsMapResources(1, &dr, 0);
            cudaArray_t da = nullptr;
            cudaGraphicsSubResourceGetMappedArray(&da, dr, 0, 0);
            cudaResourceDesc srd{}; srd.resType = cudaResourceTypeArray;
            srd.res.array.array = da;
            cudaSurfaceObject_t surf = 0;
            cudaCreateSurfaceObject(&surf, &srd);
            floatToBgraKernel << <grid, block >> > (g_oB, g_oG, g_oR, surf, g_fgW, g_fgH);
            cudaDeviceSynchronize();
            cudaDestroySurfaceObject(surf);
            cudaGraphicsUnmapResources(1, &dr, 0);
            cudaGraphicsUnregisterResource(dr);
            return 2;
        }
        return 0;
    }
}

// ==========================================
// PRESENT FRAME TO OVERLAY VIA IMGUI
// ==========================================
static void PresentToOverlay(ID3D11ShaderResourceView* srv,
    int screenW, int screenH,
    const char* fpsText)
{
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    ImGui::GetBackgroundDrawList()->AddImage(
        (ImTextureID)srv,
        ImVec2(0, 0), ImVec2((float)screenW, (float)screenH));
    ImGui::GetForegroundDrawList()->AddText(
        ImVec2(20, 20), IM_COL32(0, 255, 80, 255), fpsText);
    ImGui::GetForegroundDrawList()->AddText(
        ImVec2(20, 42), IM_COL32(255, 80, 80, 255), "[F9] to stop");

    ImGui::Render();
    const float clr[4] = { 0,0,0,1 };
    g_Ctx->OMSetRenderTargets(1, &g_OvRTV, nullptr);
    g_Ctx->ClearRenderTargetView(g_OvRTV, clr);
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    g_OvSC->Present(0, 0);
}

// ==========================================
// MAIN
// ==========================================
int main()
{
    ShowWindow(GetConsoleWindow(), SW_HIDE);
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);

    WNDCLASSEX wcUI{ sizeof(WNDCLASSEX),CS_CLASSDC,WndProcUI,0,0,
        GetModuleHandle(nullptr),nullptr,nullptr,nullptr,nullptr,
        _T("FGMUI"),nullptr };
    RegisterClassEx(&wcUI);

    WNDCLASSEX wcOv{ sizeof(WNDCLASSEX),CS_CLASSDC,WndProcOv,0,0,
        GetModuleHandle(nullptr),nullptr,nullptr,
        (HBRUSH)GetStockObject(BLACK_BRUSH),nullptr,_T("FGMOV"),nullptr };
    RegisterClassEx(&wcOv);

    g_hwndUI = CreateWindow(wcUI.lpszClassName, _T("FrameGen Master"),
        WS_OVERLAPPEDWINDOW, 100, 100, 660, 500,
        nullptr, nullptr, wcUI.hInstance, nullptr);

    g_hwndOv = CreateWindowEx(WS_EX_TOPMOST,
        wcOv.lpszClassName, nullptr, WS_POPUP,
        0, 0, screenW, screenH,
        nullptr, nullptr, wcOv.hInstance, nullptr);

    // D3D11 device — one for everything
    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL fls[] = {
        D3D_FEATURE_LEVEL_11_0,D3D_FEATURE_LEVEL_10_0 };

    DXGI_SWAP_CHAIN_DESC uisd{};
    uisd.BufferCount = 2;
    uisd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uisd.BufferDesc.RefreshRate = { 60,1 };
    uisd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    uisd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    uisd.OutputWindow = g_hwndUI;
    uisd.SampleDesc.Count = 1; uisd.Windowed = TRUE;
    uisd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    if (FAILED(D3D11CreateDeviceAndSwapChain(nullptr,
        D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, fls, 2,
        D3D11_SDK_VERSION, &uisd, &g_UiSC, &g_Dev, &fl, &g_Ctx)))
        return -1;

    {
        ID3D11Texture2D* bb;
        g_UiSC->GetBuffer(0, IID_PPV_ARGS(&bb));
        g_Dev->CreateRenderTargetView(bb, nullptr, &g_UiRTV); bb->Release();
    }

    // Overlay SwapChain — same device
    IDXGIDevice* dd = nullptr; IDXGIAdapter* da = nullptr;
    IDXGIFactory* df = nullptr;
    g_Dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dd);
    dd->GetParent(__uuidof(IDXGIAdapter), (void**)&da);
    da->GetParent(__uuidof(IDXGIFactory), (void**)&df);

    DXGI_SWAP_CHAIN_DESC ovsd{};
    ovsd.BufferCount = 2;
    ovsd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    ovsd.BufferDesc.Width = screenW; ovsd.BufferDesc.Height = screenH;
    ovsd.BufferDesc.RefreshRate = { 120,1 };
    ovsd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    ovsd.OutputWindow = g_hwndOv;
    ovsd.SampleDesc.Count = 1; ovsd.Windowed = TRUE;
    ovsd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    df->CreateSwapChain(g_Dev, &ovsd, &g_OvSC);

    {
        ID3D11Texture2D* bb;
        g_OvSC->GetBuffer(0, IID_PPV_ARGS(&bb));
        g_Dev->CreateRenderTargetView(bb, nullptr, &g_OvRTV); bb->Release();
    }
    df->Release(); da->Release(); dd->Release();

    // ImGui init on OVERLAY window
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(g_hwndOv);
    ImGui_ImplDX11_Init(g_Dev, g_Ctx);

    ShowWindow(g_hwndUI, SW_SHOWDEFAULT);
    UpdateWindow(g_hwndUI);

    bool done = false, isScaling = false;
    int  selectedWin = -1;
    HWND targetHwnd = nullptr;
    std::vector<WinInfo> winList;

    DWORD lastTick = 0;
    int origCount = 0, genCount = 0, dispOrig = 0, dispGen = 0;
    char fpsText[128] = "Warming up...";

    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        DWORD now = GetTickCount();
        if (!lastTick) lastTick = now;
        if (now - lastTick >= 1000) {
            dispOrig = origCount; dispGen = genCount;
            snprintf(fpsText, sizeof(fpsText),
                "Original: %d fps  |  Generated: %d fps  |  Total: %d fps",
                dispOrig, dispGen, dispOrig + dispGen);
            origCount = genCount = 0; lastTick = now;
        }

        // F9 to stop
        if (isScaling && (GetAsyncKeyState(VK_F9) & 0x8000)) {
            isScaling = false; targetHwnd = nullptr;
            ShowWindow(g_hwndOv, SW_HIDE);
            FreeFrameGen();
            AppLog("[System] Stopped."); Sleep(300);
        }

        // ── SCALING ───────────────────────────────────────────────────
        if (isScaling && targetHwnd && IsWindow(targetHwnd)) {
            int result = CaptureAndProcess(targetHwnd);
            if (result >= 1) {
                origCount++;
                PresentToOverlay(g_OrigSRV, screenW, screenH, fpsText);
            }
            if (result == 2) {
                genCount++;
                PresentToOverlay(g_DisplaySRV, screenW, screenH, fpsText);
            }
            continue; // skip UI render this loop
        }

        // ── UI RENDER ─────────────────────────────────────────────────
        // When not scaling, draw UI on UI window manually (no ImGui on hwndUI)
        // Use simple GDI for UI window background + ImGui renders on overlay
        // Actually: re-init ImGui for UI window temporarily
        // Simpler: just always render ImGui on UI window when not scaling

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos({ 0,0 });
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("##m", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground);

        ImGui::TextColored({ 0,1,.8f,1 }, "FrameGen Master  |  DXGI Mode");
        ImGui::Separator(); ImGui::Spacing();
        ImGui::Text("Target Window:");
        if (ImGui::Button("Refresh")) {
            RefreshWindowList(winList); selectedWin = -1;
            AppLog("[System] Refreshed.");
        }
        ImGui::Spacing();
        ImGui::BeginChild("wl", { 0,200 }, true);
        for (int i = 0; i < (int)winList.size(); i++) {
            char lbl[300];
            snprintf(lbl, sizeof(lbl), "[%p]  %s",
                (void*)winList[i].hwnd, winList[i].title);
            if (ImGui::Selectable(lbl, selectedWin == i))
                selectedWin = i;
        }
        ImGui::EndChild();
        ImGui::Spacing();

        if (selectedWin >= 0 && selectedWin < (int)winList.size()) {
            ImGui::TextColored({ .2f,1,.4f,1 }, "Selected: %s",
                winList[selectedWin].title);
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button, { .1f,.55f,.1f,1.f });
            if (ImGui::Button("START FRAME GEN", { 220,52 })) {
                targetHwnd = winList[selectedWin].hwnd;
                isScaling = true; g_firstFrame = true;
                SetupDuplication();
                ShowWindow(g_hwndOv, SW_SHOWDEFAULT);
                SetWindowPos(g_hwndOv, HWND_TOPMOST,
                    0, 0, screenW, screenH, SWP_SHOWWINDOW);
                AppLog("[OK] Started: " + std::string(winList[selectedWin].title));
                AppLog("[OK] Press F9 to stop.");
            }
            ImGui::PopStyleColor();
        }
        else {
            ImGui::TextDisabled("(Refresh then select a window)");
        }

        ImGui::Spacing(); ImGui::Separator();
        ImGui::Text("Logs:");
        ImGui::BeginChild("lg", { 0,90 }, true);
        for (auto& l : g_Logs) ImGui::TextUnformatted(l.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
            ImGui::SetScrollHereY(1.f);
        ImGui::EndChild();
        ImGui::End();

        ImGui::Render();
        const float clr[4] = { .06f,.06f,.06f,1.f };
        g_Ctx->OMSetRenderTargets(1, &g_UiRTV, nullptr);
        g_Ctx->ClearRenderTargetView(g_UiRTV, clr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_UiSC->Present(1, 0);
    }

    FreeFrameGen();
    if (g_DeskDupl) g_DeskDupl->Release();
    if (g_OvRTV)    g_OvRTV->Release();
    if (g_OvSC)     g_OvSC->Release();
    if (g_UiRTV)    g_UiRTV->Release();
    if (g_UiSC)     g_UiSC->Release();
    if (g_Ctx)      g_Ctx->Release();
    if (g_Dev)      g_Dev->Release();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    DestroyWindow(g_hwndOv);
    DestroyWindow(g_hwndUI);
    UnregisterClass(wcUI.lpszClassName, wcUI.hInstance);
    UnregisterClass(wcOv.lpszClassName, wcOv.hInstance);
    return 0;
}
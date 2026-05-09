#include <windows.h>
#include <cstdio>
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "MinHook.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// ==========================================
// CUDA Kernels
// ==========================================
__global__ void bgraToFloatKernel(cudaTextureObject_t tex,
    float* out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uchar4 p = tex2D<uchar4>(tex, x, y);
    int sz = W * H, idx = y * W + x;
    out[idx] = p.z / 255.f;
    out[sz + idx] = p.y / 255.f;
    out[2 * sz + idx] = p.x / 255.f;
}

__global__ void blendKernel(const float* a, const float* b,
    float* out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int sz = W * H, idx = y * W + x;
    out[idx] = (a[idx] + b[idx]) * 0.5f;
    out[sz + idx] = (a[sz + idx] + b[sz + idx]) * 0.5f;
    out[2 * sz + idx] = (a[2 * sz + idx] + b[2 * sz + idx]) * 0.5f;
}

__global__ void floatToBgraKernel(const float* in,
    cudaSurfaceObject_t surf, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int sz = W * H, idx = y * W + x;
    uchar4 p;
    p.x = (unsigned char)(fminf(fmaxf(in[idx], 0.f), 1.f) * 255.f);
    p.y = (unsigned char)(fminf(fmaxf(in[sz + idx], 0.f), 1.f) * 255.f);
    p.z = (unsigned char)(fminf(fmaxf(in[2 * sz + idx], 0.f), 1.f) * 255.f);
    p.w = 255;
    surf2Dwrite(p, surf, x * sizeof(uchar4), y);
}

// ==========================================
// Global State
// ==========================================
typedef HRESULT(WINAPI* PFN_Present)(IDXGISwapChain*, UINT, UINT);
typedef HRESULT(WINAPI* PFN_CreateSwapChain)(IDXGIFactory*,
    IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);

static PFN_Present         g_OrigPresent = nullptr;
static PFN_CreateSwapChain g_OrigCreateSwapChain = nullptr;

static float* g_bufA = nullptr;
static float* g_bufB = nullptr;
static float* g_bufOut = nullptr;
static int    g_W = 0, g_H = 0;
static bool   g_firstFrame = true;

static ID3D11Device* g_Dev = nullptr;
static ID3D11DeviceContext* g_Ctx = nullptr;
static ID3D11Texture2D* g_CapTex = nullptr;
static ID3D11Texture2D* g_InjectTex = nullptr;

// FPS
static DWORD g_lastTick = 0;
static int   g_origCount = 0;
static int   g_genCount = 0;
static int   g_displayOrig = 0;
static int   g_displayGen = 0;

static void FreeBuffers() {
    if (g_bufA) { cudaFree(g_bufA);       g_bufA = nullptr; }
    if (g_bufB) { cudaFree(g_bufB);       g_bufB = nullptr; }
    if (g_bufOut) { cudaFree(g_bufOut);     g_bufOut = nullptr; }
    if (g_CapTex) { g_CapTex->Release();    g_CapTex = nullptr; }
    if (g_InjectTex) { g_InjectTex->Release(); g_InjectTex = nullptr; }
    g_W = g_H = 0;
    g_firstFrame = true;
}

// ==========================================
// FPS Overlay (GDI)
// ==========================================
static void DrawFPS(HWND hwnd)
{
    DWORD now = GetTickCount();
    if (g_lastTick == 0) g_lastTick = now;
    if (now - g_lastTick >= 1000) {
        g_displayOrig = g_origCount;
        g_displayGen = g_genCount;
        g_origCount = 0;
        g_genCount = 0;
        g_lastTick = now;
    }
    if (!hwnd) return;
    HDC hdc = GetDC(hwnd);
    if (!hdc) return;
    char buf[128];
    snprintf(buf, sizeof(buf),
        "Original: %d fps  |  Generated: %d fps  |  Total: %d fps",
        g_displayOrig, g_displayGen,
        g_displayOrig + g_displayGen);
    SetBkMode(hdc, TRANSPARENT);
    RECT rs = { 21,11,800,40 }; SetTextColor(hdc, RGB(0, 0, 0));
    DrawTextA(hdc, buf, -1, &rs, DT_LEFT | DT_TOP | DT_SINGLELINE);
    RECT rg = { 20,10,800,39 }; SetTextColor(hdc, RGB(0, 255, 80));
    DrawTextA(hdc, buf, -1, &rg, DT_LEFT | DT_TOP | DT_SINGLELINE);
    ReleaseDC(hwnd, hdc);
}

// ==========================================
// Hooked Present
// ==========================================
HRESULT WINAPI HookedPresent(IDXGISwapChain* pSwapChain,
    UINT SyncInterval, UINT Flags)
{
    if (!g_Dev) {
        pSwapChain->GetDevice(__uuidof(ID3D11Device), (void**)&g_Dev);
        g_Dev->GetImmediateContext(&g_Ctx);
    }

    // FPS overlay আগে দেখাও — hook কাজ করছে কিনা confirm
    DXGI_SWAP_CHAIN_DESC scDesc{};
    pSwapChain->GetDesc(&scDesc);
    DrawFPS(scDesc.OutputWindow);

    ID3D11Texture2D* backBuf = nullptr;
    if (FAILED(pSwapChain->GetBuffer(0, IID_PPV_ARGS(&backBuf))))
        return g_OrigPresent(pSwapChain, SyncInterval, Flags);

    D3D11_TEXTURE2D_DESC desc;
    backBuf->GetDesc(&desc);
    int W = (int)desc.Width, H = (int)desc.Height;

    if (W != g_W || H != g_H) {
        FreeBuffers();
        size_t sz = (size_t)W * H * 3 * sizeof(float);
        cudaMalloc(&g_bufA, sz);
        cudaMalloc(&g_bufB, sz);
        cudaMalloc(&g_bufOut, sz);
        g_W = W; g_H = H;
        D3D11_TEXTURE2D_DESC td = desc;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        td.MiscFlags = 0;
        g_Dev->CreateTexture2D(&td, nullptr, &g_CapTex);
        g_Dev->CreateTexture2D(&td, nullptr, &g_InjectTex);
    }

    g_Ctx->CopyResource(g_CapTex, backBuf);

    cudaGraphicsResource* capRes = nullptr;
    cudaGraphicsD3D11RegisterResource(&capRes, g_CapTex,
        cudaGraphicsRegisterFlagsNone);
    cudaGraphicsMapResources(1, &capRes, 0);
    cudaArray_t capArr = nullptr;
    cudaGraphicsSubResourceGetMappedArray(&capArr, capRes, 0, 0);

    cudaResourceDesc rd{}; rd.resType = cudaResourceTypeArray;
    rd.res.array.array = capArr;
    cudaTextureDesc tdesc{};
    tdesc.addressMode[0] = tdesc.addressMode[1] = cudaAddressModeClamp;
    tdesc.filterMode = cudaFilterModePoint;
    tdesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &rd, &tdesc, nullptr);

    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    float* targetBuf = g_firstFrame ? g_bufA : g_bufB;
    bgraToFloatKernel << <grid, block >> > (texObj, targetBuf, W, H);
    cudaDeviceSynchronize();

    cudaDestroyTextureObject(texObj);
    cudaGraphicsUnmapResources(1, &capRes, 0);
    cudaGraphicsUnregisterResource(capRes);
    g_origCount++;

    if (!g_firstFrame) {
        blendKernel << <grid, block >> > (g_bufA, g_bufB, g_bufOut, W, H);
        cudaDeviceSynchronize();

        cudaGraphicsResource* injRes = nullptr;
        cudaGraphicsD3D11RegisterResource(&injRes, g_InjectTex,
            cudaGraphicsRegisterFlagsSurfaceLoadStore);
        cudaGraphicsMapResources(1, &injRes, 0);
        cudaArray_t injArr = nullptr;
        cudaGraphicsSubResourceGetMappedArray(&injArr, injRes, 0, 0);
        cudaResourceDesc srd{}; srd.resType = cudaResourceTypeArray;
        srd.res.array.array = injArr;
        cudaSurfaceObject_t surf = 0;
        cudaCreateSurfaceObject(&surf, &srd);
        floatToBgraKernel << <grid, block >> > (g_bufOut, surf, W, H);
        cudaDeviceSynchronize();
        cudaDestroySurfaceObject(surf);
        cudaGraphicsUnmapResources(1, &injRes, 0);
        cudaGraphicsUnregisterResource(injRes);

        g_Ctx->CopyResource(backBuf, g_InjectTex);
        g_OrigPresent(pSwapChain, SyncInterval, Flags);
        g_genCount++;

        g_Ctx->CopyResource(backBuf, g_CapTex);
    }

    float* tmp = g_bufA; g_bufA = g_bufB; g_bufB = tmp;
    g_firstFrame = false;
    backBuf->Release();
    return g_OrigPresent(pSwapChain, SyncInterval, Flags);
}

// ==========================================
// Hooked CreateSwapChain — এখানে VLC এর
// actual SwapChain ধরে Present hook করি
// ==========================================
HRESULT WINAPI HookedCreateSwapChain(IDXGIFactory* pFactory,
    IUnknown* pDevice, DXGI_SWAP_CHAIN_DESC* pDesc,
    IDXGISwapChain** ppSwapChain)
{
    HRESULT hr = g_OrigCreateSwapChain(pFactory, pDevice,
        pDesc, ppSwapChain);
    if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain) {
        // এই SwapChain এর vtable থেকে Present hook করো
        void** vtable = *reinterpret_cast<void***>(*ppSwapChain);
        void* presentPtr = vtable[8];
        if (!g_OrigPresent) { // একবারই hook করো
            MH_CreateHook(presentPtr, &HookedPresent,
                reinterpret_cast<void**>(&g_OrigPresent));
            MH_EnableHook(presentPtr);
        }
    }
    return hr;
}

// ==========================================
// Hook Install Thread
// ==========================================
static DWORD WINAPI InstallHookThread(LPVOID)
{
    Sleep(500);
    MH_Initialize();

    // dxgi.dll থেকে IDXGIFactory এর CreateSwapChain address বের করো
    HMODULE hDxgi = LoadLibraryA("dxgi.dll");
    if (!hDxgi) return 1;

    // IDXGIFactory বানাও vtable পেতে
    IDXGIFactory* factory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory),
        (void**)&factory)))
        return 1;

    // vtable[10] = CreateSwapChain
    void** vtable = *reinterpret_cast<void***>(factory);
    void* createSCPtr = vtable[10];
    factory->Release();

    MH_CreateHook(createSCPtr, &HookedCreateSwapChain,
        reinterpret_cast<void**>(&g_OrigCreateSwapChain));
    MH_EnableHook(createSCPtr);

    // ── Fallback: dummy SwapChain দিয়েও Present hook করো ──
    // কারণ VLC হয়তো আগেই SwapChain বানিয়ে ফেলেছে
    HWND hwnd = CreateWindowA("STATIC", "d", WS_OVERLAPPED,
        0, 0, 8, 8, nullptr, nullptr,
        GetModuleHandle(nullptr), nullptr);
    if (hwnd) {
        DXGI_SWAP_CHAIN_DESC sd{};
        sd.BufferCount = 1; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.BufferDesc.Width = 8; sd.BufferDesc.Height = 8;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.OutputWindow = hwnd; sd.SampleDesc.Count = 1;
        sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

        ID3D11Device* tmpDev = nullptr;
        ID3D11DeviceContext* tmpCtx = nullptr;
        IDXGISwapChain* tmpSC = nullptr;
        D3D_FEATURE_LEVEL fl;
        if (SUCCEEDED(D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            nullptr, 0, D3D11_SDK_VERSION, &sd,
            &tmpSC, &tmpDev, &fl, &tmpCtx)))
        {
            void** vt2 = *reinterpret_cast<void***>(tmpSC);
            void* presPtr = vt2[8];
            if (!g_OrigPresent) {
                MH_CreateHook(presPtr, &HookedPresent,
                    reinterpret_cast<void**>(&g_OrigPresent));
                MH_EnableHook(presPtr);
            }
            tmpCtx->Release(); tmpDev->Release(); tmpSC->Release();
        }
        DestroyWindow(hwnd);
    }
    return 0;
}

// ==========================================
// DllMain
// ==========================================
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hModule);
        CreateThread(nullptr, 0, InstallHookThread, nullptr, 0, nullptr);
    }
    else if (reason == DLL_PROCESS_DETACH) {
        MH_DisableHook(MH_ALL_HOOKS);
        MH_Uninitialize();
        if (g_Dev) { g_Dev->Release(); g_Dev = nullptr; }
        if (g_Ctx) { g_Ctx->Release(); g_Ctx = nullptr; }
        FreeBuffers();
    }
    return TRUE;
}
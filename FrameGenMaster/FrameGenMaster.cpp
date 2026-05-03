#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h> // Required for DirectX to CUDA memory mapping
#include <NvInfer.h>
#include <d3d11.h>
#include <dxgi1_2.h>

// Tell Visual Studio to link the D3D11 library automatically
#pragma comment(lib, "d3d11.lib") 

int main() {
    std::cout << "--- GPU Engine Initialization ---" << std::endl;

    // ==========================================
    // 1. CUDA & TensorRT Version Check
    // ==========================================
    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "[SUCCESS] CUDA Version: " << runtimeVersion << std::endl;
    std::cout << "[SUCCESS] TensorRT Version: " << getInferLibVersion() << std::endl;

    // ==========================================
    // 2. Direct3D 11 Initialization
    // ==========================================
    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    D3D_FEATURE_LEVEL featureLevel;

    std::cout << "\nAttempting to initialize Direct3D 11..." << std::endl;

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &d3dDevice,
        &featureLevel,
        &d3dContext
    );

    if (SUCCEEDED(hr)) {
        std::cout << "[SUCCESS] Direct3D 11 Device created successfully!" << std::endl;

        IDXGIDevice* pDXGIDevice = nullptr;
        if (SUCCEEDED(d3dDevice->QueryInterface(__uuidof(IDXGIDevice), (void**)&pDXGIDevice))) {
            IDXGIAdapter* pDXGIAdapter = nullptr;
            if (SUCCEEDED(pDXGIDevice->GetAdapter(&pDXGIAdapter))) {
                DXGI_ADAPTER_DESC desc;
                pDXGIAdapter->GetDesc(&desc);
                std::wcout << L"[DEBUG] D3D11 is currently using GPU: " << desc.Description << std::endl;
                pDXGIAdapter->Release();
            }
            pDXGIDevice->Release();
        }

        std::cout << "\nSetting up DXGI Desktop Duplication..." << std::endl;

        // ==========================================
        // 3. DXGI Desktop Duplication Initialization
        // ==========================================
        IDXGIDevice* dxgiDevice = nullptr;
        hr = d3dDevice->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);

        if (SUCCEEDED(hr)) {
            IDXGIAdapter* dxgiAdapter = nullptr;
            hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)&dxgiAdapter);

            if (SUCCEEDED(hr)) {
                IDXGIOutput* dxgiOutput = nullptr;
                hr = dxgiAdapter->EnumOutputs(0, &dxgiOutput);

                if (SUCCEEDED(hr)) {
                    IDXGIOutput1* dxgiOutput1 = nullptr;
                    hr = dxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dxgiOutput1);

                    if (SUCCEEDED(hr)) {
                        IDXGIOutputDuplication* desktopDuplication = nullptr;
                        hr = dxgiOutput1->DuplicateOutput(d3dDevice, &desktopDuplication);

                        if (SUCCEEDED(hr)) {
                            std::cout << "[SUCCESS] DXGI Desktop Duplication initialized!" << std::endl;

                            // ==========================================
                            // 4. The Frame Capture & CUDA Interop Loop
                            // ==========================================
                            std::cout << "\nStarting frame capture and CUDA mapping loop..." << std::endl;

                            // We will create this texture once and reuse it for all frames
                            ID3D11Texture2D* cudaFriendlyTexture = nullptr;

                            for (int i = 1; i <= 10; ++i) {
                                IDXGIResource* desktopResource = nullptr;
                                DXGI_OUTDUPL_FRAME_INFO frameInfo;

                                hr = desktopDuplication->AcquireNextFrame(500, &frameInfo, &desktopResource);

                                if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                                    std::cout << "Frame " << i << ": Timeout (Move your mouse!)" << std::endl;
                                    continue;
                                }
                                else if (FAILED(hr)) {
                                    std::cout << "Frame " << i << ": Failed to acquire. Error: " << std::hex << hr << std::endl;
                                    break;
                                }

                                ID3D11Texture2D* acquiredTexture = nullptr;
                                hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&acquiredTexture);

                                if (SUCCEEDED(hr)) {
                                    D3D11_TEXTURE2D_DESC desc;
                                    acquiredTexture->GetDesc(&desc);

                                    // -------------------------------------------------------------
                                    // STEP A: Create a clean intermediate texture (Only done once)
                                    // -------------------------------------------------------------
                                    if (cudaFriendlyTexture == nullptr) {
                                        D3D11_TEXTURE2D_DESC texDesc = desc;
                                        texDesc.Usage = D3D11_USAGE_DEFAULT;
                                        texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
                                        texDesc.CPUAccessFlags = 0;
                                        texDesc.MiscFlags = 0; // Strip away Windows OS locks/GDI flags

                                        d3dDevice->CreateTexture2D(&texDesc, nullptr, &cudaFriendlyTexture);
                                        std::cout << "-> [INIT] Created CUDA-friendly intermediate texture ("
                                            << desc.Width << "x" << desc.Height << ")\n";
                                    }

                                    // -------------------------------------------------------------
                                    // STEP B: Extremely fast VRAM-to-VRAM copy
                                    // -------------------------------------------------------------
                                    d3dContext->CopyResource(cudaFriendlyTexture, acquiredTexture);

                                    // -------------------------------------------------------------
                                    // STEP C: Map the clean texture to CUDA
                                    // -------------------------------------------------------------
                                    cudaGraphicsResource* cudaResource = nullptr;
                                    cudaError_t cudaStatus;

                                    // Notice we are registering cudaFriendlyTexture now, NOT acquiredTexture
                                    cudaStatus = cudaGraphicsD3D11RegisterResource(&cudaResource, cudaFriendlyTexture, cudaGraphicsRegisterFlagsNone);

                                    if (cudaStatus == cudaSuccess) {
                                        cudaStatus = cudaGraphicsMapResources(1, &cudaResource, 0);

                                        if (cudaStatus == cudaSuccess) {
                                            cudaArray* cuArray = nullptr;
                                            cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

                                            std::cout << "Frame " << i << ": Successfully mapped to CUDA Memory!" << std::endl;

                                            cudaGraphicsUnmapResources(1, &cudaResource, 0);
                                        }
                                        else {
                                            std::cout << "Frame " << i << ": Failed to map CUDA resource." << std::endl;
                                        }
                                        cudaGraphicsUnregisterResource(cudaResource);
                                    }
                                    else {
                                        std::cout << "Frame " << i << ": Failed to register DirectX texture to CUDA. Error Code: " << cudaStatus << std::endl;
                                    }

                                    acquiredTexture->Release();
                                }

                                desktopResource->Release();
                                desktopDuplication->ReleaseFrame();
                            }

                            // Free our intermediate texture
                            if (cudaFriendlyTexture) {
                                cudaFriendlyTexture->Release();
                            }

                            desktopDuplication->Release();
                        }
                        dxgiOutput1->Release();
                    }
                    dxgiOutput->Release();
                }
                dxgiAdapter->Release();
            }
            dxgiDevice->Release();
        }

        // ==========================================
        // 5. Global Memory Cleanup
        // ==========================================
        d3dContext->Release();
        d3dDevice->Release();
        std::cout << "\n[CLEANUP] GPU memory released safely." << std::endl;
    }

    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}
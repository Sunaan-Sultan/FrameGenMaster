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

        // --- DEBUG: Print the name of the GPU currently being used ---
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

                                    // ==========================================
                                    // 5. Map DirectX Texture to CUDA Memory (Zero-Copy)
                                    // ==========================================
                                    cudaGraphicsResource* cudaResource = nullptr;
                                    cudaError_t cudaStatus;

                                    // Register the DirectX texture with CUDA
                                    cudaStatus = cudaGraphicsD3D11RegisterResource(&cudaResource, acquiredTexture, cudaGraphicsRegisterFlagsNone);

                                    if (cudaStatus == cudaSuccess) {
                                        // Map the resource to give CUDA access
                                        cudaStatus = cudaGraphicsMapResources(1, &cudaResource, 0);

                                        if (cudaStatus == cudaSuccess) {
                                            cudaArray* cuArray = nullptr;
                                            // Get the CUDA array pointer (This is what TensorRT will use!)
                                            cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

                                            std::cout << "Frame " << i << ": Captured (" << desc.Width << "x" << desc.Height
                                                << ") & Successfully mapped to CUDA Memory!" << std::endl;

                                            // Unmap resource after we are done
                                            cudaGraphicsUnmapResources(1, &cudaResource, 0);
                                        }
                                        else {
                                            std::cout << "Frame " << i << ": Failed to map CUDA resource." << std::endl;
                                        }
                                        // Unregister resource
                                        cudaGraphicsUnregisterResource(cudaResource);
                                    }
                                    else {
                                        std::cout << "Frame " << i << ": Failed to register DirectX texture to CUDA." << std::endl;
                                    }

                                    acquiredTexture->Release();
                                }

                                desktopResource->Release();
                                desktopDuplication->ReleaseFrame();
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
        // 6. Global Memory Cleanup
        // ==========================================
        d3dContext->Release();
        d3dDevice->Release();
        std::cout << "\n[CLEANUP] GPU memory released safely." << std::endl;
    }

    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}
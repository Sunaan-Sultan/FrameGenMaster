#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <cuda_d3d11_interop.h>
#include <NvInfer.h>
#include <d3d11.h>
#include <dxgi1_2.h>

// Tell Visual Studio to link the D3D11 library automatically
#pragma comment(lib, "d3d11.lib") 

using namespace nvinfer1;

// ==========================================
// 0. The Preprocessing CUDA Kernel
// ==========================================
__global__ void bgraToFloatRgbKernel(cudaTextureObject_t inputTex, float* outputData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar4 pixel = tex2D<uchar4>(inputTex, x, y);
        int planeSize = width * height;
        int outIdx = y * width + x;

        outputData[outIdx] = pixel.z / 255.0f; // Red
        outputData[planeSize + outIdx] = pixel.y / 255.0f; // Green
        outputData[2 * planeSize + outIdx] = pixel.x / 255.0f; // Blue
    }
}

// ==========================================
// TensorRT Logger Class 
// ==========================================
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT LOG] " << msg << std::endl;
        }
    }
} gLogger;

std::vector<char> readEngineFile(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) return buffer;
    return {};
}

int main() {
    std::cout << "--- GPU Engine Initialization ---" << std::endl;

    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "[SUCCESS] CUDA Version: " << runtimeVersion << std::endl;
    std::cout << "[SUCCESS] TensorRT Version: " << getInferLibVersion() << std::endl;

    std::cout << "\nLoading AI Model (TensorRT Engine)..." << std::endl;

    std::string enginePath = "model.engine";
    std::vector<char> engineData = readEngineFile(enginePath);

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    void* buffers[3] = { nullptr, nullptr, nullptr };

    if (!engineData.empty()) {
        runtime = createInferRuntime(gLogger);
        if (runtime) {
            engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
            if (engine) {
                context = engine->createExecutionContext();
                if (context) {
                    std::cout << "[SUCCESS] AI Model Loaded & Execution Context Created!" << std::endl;

                    const int batchSize = 1;
                    const int channels = 3;
                    const int height = 1080;
                    const int width = 1920;
                    size_t frameSizeBytes = batchSize * channels * height * width * sizeof(float);

                    cudaMalloc(&buffers[0], frameSizeBytes);
                    cudaMalloc(&buffers[1], frameSizeBytes);
                    cudaMalloc(&buffers[2], frameSizeBytes);

                    context->setTensorAddress("frame_a", buffers[0]);
                    context->setTensorAddress("frame_b", buffers[1]);
                    context->setTensorAddress("generated_frame", buffers[2]);
                }
            }
        }
    }

    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    D3D_FEATURE_LEVEL featureLevel;

    std::cout << "\nAttempting to initialize Direct3D 11..." << std::endl;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &d3dDevice, &featureLevel, &d3dContext);

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
                            std::cout << "\nStarting Frame Generation Pipeline..." << std::endl;

                            ID3D11Texture2D* cudaFriendlyTexture = nullptr;

                            // -------------------------------------------------------------
                            // This flag tracks if we are capturing the very first frame
                            // -------------------------------------------------------------
                            bool isFirstFrame = true;

                            // Increased loop to 20 just to see more frames generated
                            for (int i = 1; i <= 20; ++i) {
                                IDXGIResource* desktopResource = nullptr;
                                DXGI_OUTDUPL_FRAME_INFO frameInfo;

                                hr = desktopDuplication->AcquireNextFrame(500, &frameInfo, &desktopResource);

                                if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                                    continue;
                                }
                                else if (FAILED(hr)) {
                                    break;
                                }

                                ID3D11Texture2D* acquiredTexture = nullptr;
                                hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&acquiredTexture);

                                if (SUCCEEDED(hr)) {
                                    D3D11_TEXTURE2D_DESC desc;
                                    acquiredTexture->GetDesc(&desc);

                                    if (cudaFriendlyTexture == nullptr) {
                                        D3D11_TEXTURE2D_DESC texDesc = desc;
                                        texDesc.Usage = D3D11_USAGE_DEFAULT;
                                        texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
                                        texDesc.CPUAccessFlags = 0;
                                        texDesc.MiscFlags = 0;
                                        d3dDevice->CreateTexture2D(&texDesc, nullptr, &cudaFriendlyTexture);
                                    }

                                    d3dContext->CopyResource(cudaFriendlyTexture, acquiredTexture);

                                    cudaGraphicsResource* cudaResource = nullptr;
                                    cudaError_t cudaStatus = cudaGraphicsD3D11RegisterResource(&cudaResource, cudaFriendlyTexture, cudaGraphicsRegisterFlagsNone);

                                    if (cudaStatus == cudaSuccess) {
                                        if (cudaGraphicsMapResources(1, &cudaResource, 0) == cudaSuccess) {
                                            cudaArray_t cuArray = nullptr;
                                            cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

                                            cudaResourceDesc resDesc = {};
                                            resDesc.resType = cudaResourceTypeArray;
                                            resDesc.res.array.array = cuArray;

                                            cudaTextureDesc texDesc = {};
                                            texDesc.addressMode[0] = cudaAddressModeClamp;
                                            texDesc.addressMode[1] = cudaAddressModeClamp;
                                            texDesc.filterMode = cudaFilterModePoint;
                                            texDesc.readMode = cudaReadModeElementType;
                                            texDesc.normalizedCoords = 0;

                                            cudaTextureObject_t texObj = 0;
                                            cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

                                            dim3 blockSize(16, 16);
                                            dim3 gridSize((desc.Width + blockSize.x - 1) / blockSize.x, (desc.Height + blockSize.y - 1) / blockSize.y);

                                            // -------------------------------------------------------------
                                            // STEP D: AI Inference (Pointer Swapping Logic)
                                            // -------------------------------------------------------------
                                            if (isFirstFrame) {
                                                // Store in Buffer 0 (Frame A)
                                                bgraToFloatRgbKernel << <gridSize, blockSize >> > (texObj, (float*)buffers[0], desc.Width, desc.Height);
                                                cudaDeviceSynchronize();

                                                isFirstFrame = false;
                                                std::cout << "Frame " << i << ": Captured Frame A (Waiting for Frame B...)" << std::endl;
                                            }
                                            else {
                                                // Store in Buffer 1 (Frame B)
                                                bgraToFloatRgbKernel << <gridSize, blockSize >> > (texObj, (float*)buffers[1], desc.Width, desc.Height);
                                                cudaDeviceSynchronize();

                                                // Run the AI Model!
                                                context->executeV2(buffers);
                                                cudaDeviceSynchronize();

                                                std::cout << "Frame " << i << ": AI generated an intermediate frame! (FPS Doubled)" << std::endl;

                                                // The Magic Swap: Frame B becomes Frame A for the next loop
                                                void* temp = buffers[0];
                                                buffers[0] = buffers[1];
                                                buffers[1] = temp;
                                            }
                                            // -------------------------------------------------------------

                                            cudaDestroyTextureObject(texObj);
                                            cudaGraphicsUnmapResources(1, &cudaResource, 0);
                                        }
                                        cudaGraphicsUnregisterResource(cudaResource);
                                    }
                                    acquiredTexture->Release();
                                }
                                desktopResource->Release();
                                desktopDuplication->ReleaseFrame();
                            }

                            if (cudaFriendlyTexture) cudaFriendlyTexture->Release();
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

        d3dContext->Release();
        d3dDevice->Release();
    }

    if (buffers[0]) cudaFree(buffers[0]);
    if (buffers[1]) cudaFree(buffers[1]);
    if (buffers[2]) cudaFree(buffers[2]);

    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;

    std::cout << "\n[CLEANUP] GPU memory and AI Model released safely." << std::endl;
    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}
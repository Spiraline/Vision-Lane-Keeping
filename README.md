## Vision-based Lane Keeping Module

### Requirement
- Python 3.X
- ROS
- Autoware.AI v1.14
- Vision Sensor (from ``SVL`` or ``Real Camera``)
- Actuator (from ``SVL`` or ``Real Car``)

### Installation
```
pip install opencv-python
```

## Usage with SVL Simulator
1. Download the [SVL Simulator](https://www.svlsimulator.com/) and execute `simulator`.

2. Click **Open Browser**, which will automatically show **wise.svlsimulator.com** website. Sign up and sign in with your account.

<div style="text-align:center;">
    <img src="https://user-images.githubusercontent.com/44594966/152669703-290e0d81-3327-45de-ad52-a971e02d9794.PNG" alt="svl_main" height="220"/>
    <img src="https://user-images.githubusercontent.com/44594966/153361526-bcaa5fe1-07ab-4291-8397-745666cd5932.png" alt="svl_sign_in" height="220"/>
</div>

3. Navigate to the **Clusters** tab and make your cluster.

<div style="text-align:center;">
    <img src="https://user-images.githubusercontent.com/44594966/160766172-26a89844-a7f0-4dd1-835d-578b76695335.png" alt="cluster" height="220"/>
</div>

4. Navigate to the **Vehicles** tab and click **Lexus2016RXHybrid**. Click buttons next to **Sensor Configurations**. Clone **Autoware AI** configuration to make new sensor configuration. Make sure bridge is set to **ROS**.

<div style="text-align:center;">
    <img src="https://user-images.githubusercontent.com/44594966/160766576-699b3e9c-09a0-47b4-9e0d-05afc7461507.png" alt="cluster" height="220"/>
    <img src="https://user-images.githubusercontent.com/44594966/160766948-29d2b834-4ac3-4ca9-a7d4-99ce8c13b06f.png" alt="cluster" height="220"/>
    <img src="https://user-images.githubusercontent.com/44594966/160767204-79e4c898-bb9d-4c80-b678-d589c10a21ef.png" alt="cluster" height="220"/>
    <img src="https://user-images.githubusercontent.com/44594966/160767398-e7ca4b2c-6a73-49a2-a516-e53644870032.png" alt="cluster" height="220"/>
</div>

5. Change configuration of **Main Camera** as below and save.

<div style="text-align:center;">
    <img src="https://user-images.githubusercontent.com/44594966/160768109-8667bcb7-e395-449e-9c8a-81eb0a984b35.png" alt="cluster" height="400"/>
</div>

6. Navigate to the **Simulations** tab and click **Add New** button. Fill the blank as below.
    - Simulation Name: As your wish
    - Cluster: a cluster you made in **3**
    - Runtime Template: Random Traffic
    - Map: CubeTown
    - Vehicle: Lexus2016RXHybrid (with configuration you made in **4**)
    - Autopilot: Autoware.AI v1.14

<div style="text-align:center;">
    <img src="https://user-images.githubusercontent.com/44594966/160768668-7be761c7-63a3-4d6d-8e85-8b7799147193.png" alt="cluster" height="220"/>
    <img src="https://user-images.githubusercontent.com/44594966/160769391-951883b4-7b18-4fa3-a16a-d5956ed253ce.png" alt="cluster" height="220"/>
</div>

7. Run rosbridge node
    ```
    roslaunch rosbridge_server rosbridge_websocket.launch
    ```

8. Click **Run Simulation** and click play icon on simulator. If the count of topic increases in bridge tab (plug icon), SVL connected with bridge successfully.

<div style="text-align:center;">
    <img src="https://user-images.githubusercontent.com/44594966/160772542-93f6ff64-9e4e-46a5-8681-748ed94e245c.png" alt="cluster" height="220"/>
    <img src="https://user-images.githubusercontent.com/44594966/160772743-3cabb046-2705-4ddf-9c32-8bee8301f619.png" alt="cluster" height="220"/>
</div>

9. Copy `config/config-svl.yaml` to `config/config`.

10. Run LKAS node.

    ```
    python lkas.py
    roslaunch twist_filter twist_filter.launch
    ```

## Demo
![lkas_success](https://user-images.githubusercontent.com/44594966/150334917-fb741128-8fbb-4e73-944e-353a8ca5f5d3.gif)

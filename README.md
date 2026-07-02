     " Universal In-Silico Neuroinflammation Simulator "
 [ Click Here to Launch the Live Web Application](https://universal-neuroinflammation-simulator-qebvnrydyxvbpxavfybq4u.streamlit.app/)

  -Scientific Overview
This framework is an open-source **In-Silico Pathokinetic Dashboard**. It bridges environmental climate stress with cellular neuroscience by pulling live, authentic satellite reanalysis archives from the **Copernicus ERA5 proxy**. 

The backend architecture uses a system of **Ordinary Differential Equations (ODEs)** to calculate the dynamic kinetic rates of Blood-Brain Barrier ($BBB$) degradation and subsequent Microglial ($M1$) activation in real-time for any coordinate on Earth.

  -Tech Stack
* **Frontend UI:** Streamlit Cloud
* **Mathematical Solver:** SciPy (`odeint`)
* **Data Ingestion:** REST APIs (Open-Meteo Copernicus Archive)
* **Data Analytics:** Pandas, NumPy, Seaborn, Matplotlib

   -License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

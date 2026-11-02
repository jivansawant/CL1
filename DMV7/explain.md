# DMV Practical 7 — Interacting with Web APIs

## 1. CORE TAKEAWAY

**This practical teaches you how to fetch live weather data from a web API, store it in a Pandas DataFrame, calculate basic weather statistics, and visualize weather patterns across cities.**

The main concept is that instead of downloading a dataset manually, we use an API to retrieve data directly from an online service.

**One important limitation in your original code:** it retrieves the current weather for five cities, not weather history over time. Therefore, we can compare cities, but we cannot calculate daily or monthly trends from this data alone.

## 2. MINIMALIST PYTHON CODE

We'll keep the five cities and the important operations from your original code, but remove unnecessary plotting options and repetitive code.

**Before running the code:** Obtain your API key from [OpenWeatherMap](https://openweathermap.org/api?utm_source=chatgpt.com). Your original code contains a hardcoded API key. If that is a real key, replace it and keep the new key private.

### Notebook Cell 1 — Python Code

Copy this into your first Python cell.

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

API_KEY = "YOUR_API_KEY"
cities = ["Pune", "Mumbai", "Delhi", "Bangalore", "Chennai"]
weather = []

# Fetch weather data
for city in cities:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        weather.append({
            "City": city,
            "Temperature": data["main"]["temp"],
            "Humidity": data["main"]["humidity"],
            "Wind_Speed": data["wind"]["speed"],
            "Precipitation": data.get("rain", {}).get("1h", 0),
            "Latitude": data["coord"]["lat"],
            "Longitude": data["coord"]["lon"]
        })

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Error fetching {city}: {e}")

# Create and clean DataFrame
df = pd.DataFrame(weather).dropna()

print("Weather Data:")
display(df)

# Calculate statistics
print("\nAverage Temperature:", df["Temperature"].mean())
print("Maximum Temperature:", df["Temperature"].max())
print("Minimum Temperature:", df["Temperature"].min())
print("Average Humidity:", df["Humidity"].mean())
print("Average Wind Speed:", df["Wind_Speed"].mean())

# Temperature comparison
df.plot(x="City", y="Temperature", kind="bar", legend=False)
plt.ylabel("Temperature (°C)")
plt.title("Temperature Comparison")
plt.tight_layout()
plt.show()

# Temperature vs Humidity
plt.scatter(df["Temperature"], df["Humidity"])
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("Temperature vs Humidity")
plt.show()

# Correlation heatmap
corr = df[["Temperature", "Humidity", "Wind_Speed", "Precipitation"]].corr()
sns.heatmap(corr, annot=True)
plt.title("Weather Correlation")
plt.show()

# Geographic visualization
plt.scatter(df["Longitude"], df["Latitude"])

for _, row in df.iterrows():
    plt.text(row["Longitude"], row["Latitude"], row["City"])

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("City Locations")
plt.show()
```

**Before execution:**
- Replace `"YOUR_API_KEY"` with your actual API key.
- Install missing libraries if necessary: `pip install requests pandas matplotlib seaborn`.
- Run the code in Jupyter Notebook, since it uses `display(df)`.

---

## 3. WHY CODE WALKTHROUGH

Copy the following into your **second notebook cell**, choosing Markdown as the cell type.

### 1. Import libraries

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

- `requests`: Sends HTTP requests to the weather API.
- `pandas`: Stores and analyzes the retrieved data.
- `matplotlib`: Creates graphs.
- `seaborn`: Creates the correlation heatmap.

We import only the libraries required for this practical.

### 2. Set the API key and cities

```python
API_KEY = "YOUR_API_KEY"
cities = ["Pune", "Mumbai", "Delhi", "Bangalore", "Chennai"]
weather = []
```

- `API_KEY` authenticates our requests.
- `cities` contains the locations whose weather we want to retrieve.
- `weather` is an empty list that will store the results.

We use a list because we need to collect the results for multiple cities before creating a DataFrame.

### 3. Fetch data from the API

```python
response = requests.get(url, timeout=15)
response.raise_for_status()
data = response.json()
```

**Why do we do this?**

1. `requests.get()` sends a GET request to OpenWeatherMap.
2. `timeout=15` prevents the request from waiting indefinitely.
3. `raise_for_status()` raises an exception if the server returns an HTTP error status.
4. `response.json()` converts the API response into Python data structures, usually dictionaries.

The URL contains `units=metric`, so the temperature is returned in Celsius instead of Kelvin.

**Important:** If the API key is invalid or the request fails, the exception handler reports the error instead of allowing that request to terminate the entire loop.

### 4. Extract weather attributes

```python
weather.append({
    "City": city,
    "Temperature": data["main"]["temp"],
    "Humidity": data["main"]["humidity"],
    "Wind_Speed": data["wind"]["speed"],
    "Precipitation": data.get("rain", {}).get("1h", 0),
    "Latitude": data["coord"]["lat"],
    "Longitude": data["coord"]["lon"]
})
```

The API returns a JSON response containing different categories of information.

| Attribute | Meaning | Unit |
|---|---|---|
| `City` | City name | — |
| `Temperature` | Current temperature | °C |
| `Humidity` | Relative humidity | % |
| `Wind_Speed` | Wind speed | m/s |
| `Precipitation` | Rainfall reported for the last hour | mm |
| `Latitude` | Geographic north–south coordinate | Degrees |
| `Longitude` | Geographic east–west coordinate | Degrees |

**Why use `data.get("rain", {}).get("1h", 0)`?**

The API might not include the `rain` field when no rainfall is reported.

- `data.get("rain", {})` returns the rainfall dictionary if available, or an empty dictionary otherwise.
- `.get("1h", 0)` retrieves the one-hour rainfall value or uses `0` when that value is absent.

This avoids a missing-key error.

Remember that a missing rainfall field is treated as zero here. This is a simplifying assumption, not a guarantee that missing data always means no rain.

### 5. Create and clean the DataFrame

```python
df = pd.DataFrame(weather).dropna()
display(df)
```

`pd.DataFrame(weather)` converts the list of dictionaries into a table.

Each row represents a city, and each column represents a weather attribute.

`dropna()` removes rows containing missing values.

We do not use `fillna(0)` on every column because replacing a missing temperature or humidity with zero could produce misleading statistics.

### 6. Calculate weather statistics

```python
df["Temperature"].mean()
df["Temperature"].max()
df["Temperature"].min()
df["Humidity"].mean()
df["Wind_Speed"].mean()
```

These operations summarize the retrieved weather data.

- `mean()` calculates the average.
- `max()` finds the largest value.
- `min()` finds the smallest value.

For example, the average temperature gives us the mean of the current temperatures retrieved for all successfully processed cities.

**This is a comparison across cities, not an average over time.**

### 7. Plot temperature comparison

```python
df.plot(x="City", y="Temperature", kind="bar", legend=False)
```

A bar chart makes it easy to compare current temperatures across cities.

- X-axis: City names.
- Y-axis: Temperature in Celsius.
- `kind="bar"` selects a bar chart.
- `legend=False` removes an unnecessary legend because there is only one plotted variable.

### 8. Plot temperature versus humidity

```python
plt.scatter(df["Temperature"], df["Humidity"])
```

A scatter plot helps us examine the relationship between two numerical variables.

- X-axis: Temperature.
- Y-axis: Humidity.
- Each point represents one city.

The plot helps us observe whether temperature and humidity appear related in this small sample.

It does not establish causation, and five cities are insufficient for drawing strong conclusions about general weather patterns.

### 9. Create a correlation heatmap

```python
corr = df[["Temperature", "Humidity", "Wind_Speed", "Precipitation"]].corr()
sns.heatmap(corr, annot=True)
```

First, `.corr()` calculates the pairwise Pearson correlation coefficients between the selected numerical columns.

Then, `sns.heatmap()` displays the correlation matrix visually.

- `annot=True` prints the correlation values inside the cells.
- Positive values indicate a positive linear relationship.
- Negative values indicate a negative linear relationship.
- Values near zero indicate little or no linear relationship.

**Important:** Correlation does not prove causation. Also, with very few cities, these correlation values can be unstable or misleading.

### 10. Visualize geographic locations

```python
plt.scatter(df["Longitude"], df["Latitude"])
```

Latitude and longitude represent the geographic coordinates of each city.

The scatter plot shows the relative positions of the cities.

We then use:

```python
for _, row in df.iterrows():
    plt.text(row["Longitude"], row["Latitude"], row["City"])
```

This labels each point with its city name.

This is a basic geographic visualization, not a full geographic map. It does not display coastlines, borders, or geographic features.

---

## 4. VIVA CONCEPT CRASH COURSE

### A. What is an API?

An **API (Application Programming Interface)** allows two software applications to communicate with each other.

In this practical, our Python program requests weather information from OpenWeatherMap.

Instead of collecting weather data manually, we retrieve it programmatically.

### B. What is a Web API?

A Web API allows applications to exchange data over a network, typically using HTTP.

Our program sends a request to a remote server and receives a response.

### C. What is a REST API?

A REST API follows the principles of Representational State Transfer.

REST APIs commonly use HTTP methods such as:

- `GET`: Retrieve data.
- `POST`: Submit data.
- `PUT` or `PATCH`: Update data.
- `DELETE`: Delete data.

Our practical uses the **GET method** because we only need to retrieve weather information.

### D. What is an API key?

An API key is a credential used by a service to identify or authorize an application making requests.

We include our OpenWeatherMap API key in the request URL.

Never publish a real API key in publicly shared code.

### E. What is JSON?

JSON stands for **JavaScript Object Notation**.

It is a common format for exchanging structured data between applications.

Example:

```json
{
    "main": {
        "temp": 28.5,
        "humidity": 65
    }
}
```

Python accesses these nested values using dictionary keys:

```python
data["main"]["temp"]
data["main"]["humidity"]
```

### F. What is data preprocessing?

Data preprocessing prepares raw data for analysis.

Common operations include:

- Handling missing values.
- Removing duplicate records.
- Correcting inconsistent formats.
- Converting data types when necessary.

In this practical, we use `dropna()` to remove rows containing missing values.

We do not need a complicated cleaning pipeline because the API already returns structured data.

### G. What is data modeling in this practical?

Here, data modeling refers to analyzing the retrieved data using summary statistics and relationships.

Examples include calculating averages, finding maximum and minimum temperatures, and examining correlations.

**We are not training a machine-learning model in this practical.** The listed tasks can be satisfied using descriptive statistical analysis.

### H. What is data aggregation?

Data aggregation combines multiple observations into summary values.

For example, calculating the average temperature across five cities is an aggregation.

Pandas commonly uses:

```python
df["Temperature"].mean()
```

However, daily, monthly, and seasonal aggregation requires observations collected across those time periods. Our current-weather API endpoint provides a snapshot, not a historical time series.

### I. What is correlation?

Correlation measures the strength and direction of the relationship between two variables.

Pearson's correlation coefficient, usually represented by `r`, ranges from `-1` to `+1`.

| Value | Interpretation |
|---|---|
| `+1` | Perfect positive linear correlation |
| `0` | No linear correlation |
| `-1` | Perfect negative linear correlation |

A correlation of `+0.8` indicates a strong positive linear relationship, while `-0.8` indicates a strong negative linear relationship.

Correlation alone does not establish cause and effect.

### J. What is geospatial visualization?

Geospatial visualization represents information associated with geographic locations.

In this practical, we use latitude and longitude to plot city locations.

A true map-based weather visualization would require geographic boundaries or a mapping library, which is unnecessary for this simplified version.

### K. What is the difference between current and historical weather data?

This is an important distinction for your viva.

| Current weather API | Historical weather API |
|---|---|
| Provides a current snapshot | Provides observations from past periods |
| Useful for comparing cities now | Useful for studying changes over time |
| Does not establish daily trends by itself | Can support daily, monthly, and seasonal analysis |

Our code uses the current-weather endpoint. If the examiner asks about weather trends over time, explain that we would need historical observations or data collected repeatedly over time.

---

## 5. TOP 3 EXPECTED VIVA QUESTIONS

**Q1. What is an API, and why is it used in this practical?**

**Answer:** An API allows different software applications to communicate. We use the OpenWeatherMap API to retrieve current weather data programmatically.

**Q2. Why do we convert the API response into a Pandas DataFrame?**

**Answer:** A DataFrame organizes the retrieved data into rows and columns, making it easy to clean, calculate statistics, and visualize weather attributes.

**Q3. What is a correlation heatmap, and what does correlation tell us?**

**Answer:** A correlation heatmap visually represents the relationships between numerical variables. Correlation measures the direction and strength of a relationship, but it does not prove causation.

---

### Final exam tip

Remember this sequence:

**API request → JSON response → Extract attributes → DataFrame → Clean data → Calculate statistics → Visualize results.**

One final detail: the practical statement mentions analyzing weather patterns over time, but the supplied endpoint only retrieves current conditions. Your simplified code correctly analyzes and visualizes the current weather across cities; historical analysis would require historical data or repeated observations.
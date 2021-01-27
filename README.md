# cira-boilerplate
Boilerplate for the cira projects, so that you can get stated creating you trading algorithm!

## Installation

Copy the repository using the [github "Use this template"](https://github.com/cira-group/cira-boilerplate/generate) or [clone](git@github.com:cira-group/cira-boilerplate.git) the repository

When you have it downloaded you can start installing it.
On macOS and Linux:

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Getting started

For more documentation check out the cira **[wiki](https://github.com/AxelGard/cira/wiki)**.

Since the Alpaca trade API need a API key, you need to generate your own key at [alpaca markets website](https://app.alpaca.markets/signup). If you want to play around with it you can try paper trading (recommended for beginners). I recommend keep it in a **JSON file** which cira needs the **path** to.
You can also set the variables directly or use an environment variable, see the [wiki](https://github.com/AxelGard/cira/wiki/Storing-the-Alpaca-API-key) for diffrent the ways. However, it is recommended that you store it in a file just make sure not to upload that file on any public repositories. <br>
**key.json**
```json
{
  "APCA-API-KEY-ID":"your_pub_key",
  "APCA-API-SECRET-KEY":"your_private_key"
}
```

then you can start using the lib
```python
import cira
cira.alpaca.KEY_FILE = "../mypath/key.json"
stock = cira.Stock("TSLA")
stock.buy(1)
stock.sell(1)
```

New classes with cira v.2!
```python
portfolio = cira.Portfolio() # methods for your portfolio
exchange = cira.Exchange() # methods for exchange
stock = cira.Stock("TSLA") # a class for one stock
```

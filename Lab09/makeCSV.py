import pandas as pd

def main():

    dates = []
    expiry = []
    strike = []
    call_price = []
    put_price = []

    for i in range(1,12):
        call_name = str(i) + ".csv"
        put_name = str(i) + "p.csv"

        df_call = pd.read_csv(call_name, usecols=['Date', 'Expiry', 'Strike Price', 'Settle Price'])
        df_put = pd.read_csv(put_name, usecols=['Settle Price'])

        dates.extend(list(df_call['Date']))
        expiry.extend(list(df_call['Expiry']))
        strike.extend(list(df_call['Strike Price']))
        call_price.extend(list(df_call['Settle Price']))
        put_price.extend(list(df_put['Settle Price']))

    data = {'Date': dates, 'Expiry': expiry, 'Strike Price': strike, 'Call Price': call_price, 'Put Price': put_price}

    df = pd.DataFrame(data) 
    df.to_csv('stockoptiondata_COALINDIA.csv', header=True, index=False) 
        
if __name__ == '__main__':
    main()

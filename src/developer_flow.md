# Feature Design Flow
1. Find what can be parsed
2. Decide the interface and behaviour with existing features

# Feature Adding Flow
1. Add flag in **CLI**
    1. Add flag as a parser argument
    2. Adjust existing cli logic in presence of the flag
2. Let **Context** validate it
3. Implement: 
   1. **Parser**
   2. **Scrapper**
   3. **Scrap Mgr**:
      1. Scrapping Method
      2. Scrap Kind
      3. Conditions and order of execution in main method
   4. **App Mgr** behaviour
4. ...
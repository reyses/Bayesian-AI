# RUNG 3 — vision chart classify | model=gemma4:e2b | n=10

                              file |       truth |        pred | ok
------------------------------------------------------------------------------
              trade_00019_CHOP.png |        CHOP |  CLEAN_RIDE | n
         trade_00118_SMALL_WIN.png |   SMALL_WIN |  CLEAN_RIDE | n
         trade_00200_GAVE_BACK.png |   GAVE_BACK |  CLEAN_RIDE | n
        trade_00208_CLEAN_RIDE.png |  CLEAN_RIDE |  CLEAN_RIDE | Y
              trade_00235_CHOP.png |        CHOP |  CLEAN_RIDE | n
         trade_00263_SMALL_WIN.png |   SMALL_WIN |  CLEAN_RIDE | n
        trade_00378_CLEAN_RIDE.png |  CLEAN_RIDE |  CLEAN_RIDE | Y
         trade_00581_GAVE_BACK.png |   GAVE_BACK |  CLEAN_RIDE | n
        trade_00774_CLEAN_RIDE.png |  CLEAN_RIDE |  CLEAN_RIDE | Y
        trade_00822_SMALL_LOSS.png |  SMALL_LOSS |  CLEAN_RIDE | n

ARCHETYPE accuracy: 3/10 = 30%
'read the chart' (mentions direction): 10/10 = 100%
JSON-format failures: 0/10 = 0%
latency/call: mean 2.0s

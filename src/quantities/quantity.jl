"""create a custum data type called AbstractQuantity: 
    - unified interface for all quantities (no changes in pipeline required 
      if more quantities are added later)
    - compute! can be used for all quantities (depending on the specifit type 
      the correct logic will be later implemented)     
"""
abstract type AbstractQuantity end
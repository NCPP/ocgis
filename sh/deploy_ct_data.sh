## climatetranslator

## move data to dev-hydra
scp -v ocgis_data.tar.gz ben.koziol@dev-hydra.esrl.svc:/home/local/WX/ben.koziol
scp -v ~/htmp/ocgis_geometries.json ben.koziol@dev-hydra.esrl.svc:/home/local/WX/ben.koziol

## move data to hydra
scp -v ocgis_data.tar.gz ben.koziol@hydra.fsl.noaa.gov:/home/ben.koziol
scp -v ~/htmp/ocgis_geometries.json ben.koziol@hydra.fsl.noaa.gov:/home/ben.koziol
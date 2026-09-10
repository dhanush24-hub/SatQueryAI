import type {IsroAnalysisData,TelemetryMetadata,LulcMetric,SpectralAnalysis,Finding,EvidenceItem} from './domain';

export function getIsroData(demoId:string|null,scenes:{name:string;label?:string}[],findings:Finding[]=[],evidenceItems:EvidenceItem[]=[]):IsroAnalysisData | undefined{
 if(demoId==='urban'){
  return {
   confidenceScore:94,
   telemetry:[
    {platform:'Landsat 5',sensor:'Thematic Mapper (TM)',resolutionGsd:'30 m (Multispectral)',orbitPass:'WRS-2 Path 176 / Row 39 (Desc)',solarElevation:'61.2°',solarAzimuth:'109.8°',crs:'EPSG:32636 (WGS 84 / UTM Zone 36N)',coregistrationRms:'0.28 px'},
    {platform:'Landsat 8',sensor:'Operational Land Imager (OLI)',resolutionGsd:'30 m (15 m Panchromatic)',orbitPass:'WRS-2 Path 176 / Row 39 (Desc)',solarElevation:'58.4°',solarAzimuth:'128.2°',crs:'EPSG:32636 (WGS 84 / UTM Zone 36N)',coregistrationRms:'0.21 px'}
   ],
   lulc:[
    {category:'Built-up / Urban Sprawl',color:'#f59e0b',t1Percent:18.2,t2Percent:36.8,deltaPercent:18.6,areaHectares:4.2,trend:'increase'},
    {category:'Vegetated Agriculture',color:'#10b981',t1Percent:34.5,t2Percent:29.1,deltaPercent:-5.4,areaHectares:1.8,trend:'decrease'},
    {category:'Open Desert / Bare Soil',color:'#eab308',t1Percent:45.1,t2Percent:32.3,deltaPercent:-12.8,areaHectares:2.9,trend:'decrease'},
    {category:'River Nile & Waterways',color:'#06b6d4',t1Percent:2.2,t2Percent:1.8,deltaPercent:-0.4,areaHectares:0.1,trend:'stable'}
   ],
   spectral:[
    {indexName:'NDBI (Normalized Difference Built-up)',description:'SWIR1 - NIR / SWIR1 + NIR',valueT1:'-0.18',valueT2:'+0.34',deltaInterpretation:'+0.52 surge in eastern desert grid confirms high-density concrete and asphalt construction.'},
    {indexName:'NDVI (Vegetation Density)',description:'NIR - Red / NIR + Red',valueT1:'+0.48',valueT2:'+0.39',deltaInterpretation:'Northern agricultural perimeter indicates seasonal crop rotation rather than irreversible loss.'},
    {indexName:'MNDWI (Modified Water Index)',description:'Green - SWIR / Green + SWIR',valueT1:'-0.65',valueT2:'-0.62',deltaInterpretation:'Nile river boundary remained structurally stable with minimal hydrologic shift.'}
   ],
   changeDetectionMask:[
    {x:63,y:43,width:29,height:31,intensity:0.94,category:'urban_expansion'},
    {x:3,y:3,width:29,height:25,intensity:0.65,category:'vegetation_shift'}
   ]
  };
 }

 if(demoId==='flood'){
  return {
   confidenceScore:78,
   telemetry:[
    {platform:'Landsat 8',sensor:'OLI False-Color 6-5-4 (SWIR-NIR-Red)',resolutionGsd:'30 m (SWIR Resampled)',orbitPass:'WRS-2 Path 14 / Row 36',solarElevation:'52.1°',solarAzimuth:'142.3°',crs:'EPSG:32618 (WGS 84 / UTM Zone 18N)',coregistrationRms:'0.35 px'},
    {platform:'Sentinel-1A',sensor:'C-band SAR (Synthetic Aperture Radar)',resolutionGsd:'10 m (Ground Range Detected)',orbitPass:'Relative Orbit 148 (Descending)',incidenceAngle:'39.4° (VV Polarisation)',crs:'EPSG:4326 (WGS 84 Geographic)',coregistrationRms:'1.42 px (Domain Gap)'}
   ],
   lulc:[
    {category:'Open Inundation / Water',color:'#06b6d4',t1Percent:8.4,t2Percent:24.6,deltaPercent:16.2,areaHectares:14.5,trend:'increase'},
    {category:'Saturated Soil / Floodplain',color:'#a855f7',t1Percent:12.1,t2Percent:22.8,deltaPercent:10.7,areaHectares:9.8,trend:'increase'},
    {category:'Forest / Emergent Canopy',color:'#10b981',t1Percent:48.2,t2Percent:36.4,deltaPercent:-11.8,areaHectares:10.6,trend:'decrease'},
    {category:'Infrastructure & High Ground',color:'#f59e0b',t1Percent:31.3,t2Percent:16.2,deltaPercent:-15.1,areaHectares:13.7,trend:'decrease'}
   ],
   spectral:[
    {indexName:'MNDWI (Water Inundation Index)',description:'Green - SWIR1 / Green + SWIR1',valueT1:'-0.32',valueT2:'+0.58',deltaInterpretation:'Extreme water absorption in SWIR confirms severe localized standing flood along the Trent River.'},
    {indexName:'SAR Backscatter σ° (VV Polarisation)',description:'C-band Radar Surface Roughness',valueT1:'-8.5 dB',valueT2:'-18.2 dB',deltaInterpretation:'Specular radar scattering over smooth floodwaters creates -9.7 dB backscatter attenuation.'},
    {indexName:'Double-Bounce Radar Interaction',description:'Tree Trunk / Floodwater Interface',valueT1:'+2.1 dB',valueT2:'+6.8 dB',deltaInterpretation:'Strong dihedral reflection from flooded forest stands in Sentinel-1 SAR acquisition.'}
   ],
   changeDetectionMask:[
    {x:33,y:21,width:43,height:48,intensity:0.89,category:'water_inundation'},
    {x:30,y:8,width:39,height:45,intensity:0.75,category:'radar_flood_proxy'}
   ]
  };
 }

 if(demoId==='single'){
  return {
   confidenceScore:91,
   telemetry:[
    {platform:'Landsat 7',sensor:'Enhanced Thematic Mapper Plus (ETM+)',resolutionGsd:'30 m (Multispectral)',orbitPass:'Mosaic Path 137 / Row 45',solarElevation:'48.6°',solarAzimuth:'138.4°',crs:'EPSG:32645 (WGS 84 / UTM Zone 45N)',coregistrationRms:'0.18 px'}
   ],
   lulc:[
    {category:'Dense Mangrove Canopy',color:'#10b981',t1Percent:58.6,t2Percent:58.6,deltaPercent:0,areaHectares:42.1,trend:'stable'},
    {category:'Tidal Estuary & Waterways',color:'#06b6d4',t1Percent:28.4,t2Percent:28.4,deltaPercent:0,areaHectares:20.4,trend:'stable'},
    {category:'Intertidal Mudflats / Sediment',color:'#eab308',t1Percent:13.0,t2Percent:13.0,deltaPercent:0,areaHectares:9.3,trend:'stable'}
   ],
   spectral:[
    {indexName:'NDVI (Mangrove Vigor Index)',description:'NIR - Red / NIR + Red',valueT1:'+0.62',valueT2:'+0.62',deltaInterpretation:'High canopy density throughout the central delta reserve with minimal anthropogenic fragmentation.'},
    {indexName:'NDWI (Gao Water Index)',description:'NIR - SWIR / NIR + SWIR',valueT1:'+0.44',valueT2:'+0.44',deltaInterpretation:'High moisture content and deep tidal penetration through dendritic channels.'}
   ],
   changeDetectionMask:[
    {x:28,y:31,width:38,height:39,intensity:0.82,category:'tidal_channels'},
    {x:40,y:46,width:35,height:39,intensity:0.91,category:'vegetated_delta'}
   ]
  };
 }

 // CRITICAL INTEGRITY CHECK: Never fabricate satellite platform telemetry,
 // LULC change deltas, spectral indices, or fake confidence scores on user-uploaded imagery.
 // Synthetic telemetry is strictly confined to curated demonstration fixtures.
 return undefined;
}

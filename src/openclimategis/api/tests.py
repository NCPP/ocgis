from django.test import TestCase, TransactionTestCase
from django.test.client import Client
from climatedata.models import NetcdfDataset
import unittest
import itertools
from util.helpers import reverse_wkt, get_temp_path


#def disabled(f):
#    warn('{0} TEST DISABLED!'.format(f.__name__))

#class TestViews(TestCase):
#    
#    fixtures = ['test_usgs-cida-maurer.json']
#    
#    def test_get_choices(self):
#        choices = get_choices(Archive)
#        self.assertEqual(len(choices),2)
#        
#        choices = get_choices(SimulationOutput,'pk','run',True)
#        self.assertEqual(len(choices),3)
    

class TestUrls(TestCase):
    """Test URLs for correct response codes."""
    
#    fixtures = ['luca_fixtures.json']
    fixtures = ['test_usgs-cida-maurer.json']
    
    def setUp(self):
        self.client = Client()
    
    def test_fixture_loading(self):
        '''Check that the test fixture loaded correctly'''
        self.assertEqual(NetcdfDataset.objects.count(), 1)
    
    def test_api_html(self):
        '''Creates an HTML representation of the API main page'''
        response = self.client.get('/api/')
        if response.status_code != 200:
                print response.content
        self.assertEqual(response.status_code, 200)
        
    def test_multipolygon_url(self):
        wkt = 'MULTIPOLYGON (((-88.497527278807524 48.17379537233009,-88.625327278926548 48.033167372199117,-88.901547279183802 47.960248372131204,-89.028622279302155 47.850655372029138,-89.139885279405775 47.824076372004384,-89.192916279455162 47.844613372023517,-89.201787279463417 47.883857372060064,-89.156099279420872 47.939228372111629,-88.497527278807524 48.17379537233009)),((-88.500681278810461 47.290180371507155,-88.437901278751994 47.355896371568363,-88.211392278541041 47.44783537165398,-87.788120278146849 47.470793371675363,-87.70438327806886 47.415950371624291,-87.737510278099705 47.393024371602934,-87.917042278266905 47.358007371570324,-88.222279278551184 47.200752371423874,-88.412843278728658 46.988094371225813,-88.470664278782507 47.111472371340724,-88.594262278897617 47.134765371362413,-88.595632278898904 47.243593371463767,-88.500681278810461 47.290180371507155)),((-85.859844276350998 45.969469370277146,-85.914955276402324 45.957978370266446,-85.917104276404316 45.918192370229391,-86.067891276544756 45.964210370272255,-86.259319276723033 45.946929370256157,-86.315638276775488 45.90568237021774,-86.343795276801714 45.834396370151353,-86.458275276908324 45.762747370084625,-86.529390276974553 45.748961370071783,-86.522010276967691 45.724094370048626,-86.576124277018081 45.710174370035659,-86.629784277068055 45.621233369952826,-86.685053277119536 45.650048369979665,-86.696919277130576 45.692511370019211,-86.584735277026098 45.813879370132241,-86.761469277190699 45.826067370143591,-86.901624277321233 45.714778370039951,-87.123759277528109 45.696246370022692,-87.260707277655655 45.554802369890957,-87.332227277722254 45.423942369769087,-87.583864277956621 45.162733369525817,-87.592514277964668 45.108501369475306,-87.672814278039453 45.140672369505268,-87.729669278092402 45.176604369538737,-87.736200278098494 45.199072369559659,-87.721628278084921 45.211672369571396,-87.719668278083091 45.23677136959477,-87.705142278069559 45.247086369604375,-87.704471278068937 45.27220536962777,-87.645362278013891 45.348169369698518,-87.64368427801233 45.361856369711262,-87.689598278055087 45.391269369738652,-87.760038278120689 45.352897369702916,-87.828008278183987 45.358321369707973,-87.84128227819636 45.346149369696633,-87.862096278215745 45.370165369719004,-87.868535278221742 45.372072369720783,-87.873974278226797 45.36208536971148,-87.883610278235778 45.365854369714988,-87.84953127820404 45.406117369752486,-87.860267278214039 45.445098369788788,-87.81361427817059 45.466460369808686,-87.789385278148018 45.499067369839054,-87.805141278162694 45.544525369881384,-87.828602278184547 45.5685913699038,-87.786312278145161 45.568519369903733,-87.775075278134693 45.600387369933415,-87.776045278135598 45.613200369945346,-87.81993827817648 45.654450369983763,-87.817054278173785 45.665390369993951,-87.780945278140166 45.67591537000375,-87.777473278136924 45.684101370011376,-87.801156278158984 45.701324370027422,-87.801553278159346 45.711391370036793,-87.842362278197356 45.722418370047066,-87.873629278226474 45.750699370073406,-87.969179278315465 45.766448370088071,-87.990070278334926 45.795046370114704,-88.051639278392258 45.78611237010638,-88.088734278426813 45.791532370111426,-88.12994927846519 45.819402370137382,-88.121786278457591 45.834878370151799,-88.065421278405097 45.873642370187902,-88.095764278433364 45.891803370204812,-88.093850278431574 45.920615370231644,-88.111390278447914 45.926287370236935,-88.150438278484273 45.936293370246247,-88.180194278511991 45.953516370262292,-88.214992278544401 45.947901370257057,-88.257168278583677 45.967055370274899,-88.299152278622785 45.961944370270139,-88.321323278643433 45.966712370274578,-88.369938278688707 45.994587370300536,-88.403522278719976 45.983422370290143,-88.454319278767287 46.000760370306288,-88.483814278794753 45.999151370304787,-88.494083278804325 46.012960370317657,-88.515613278824375 46.018609370322913,-88.548358278854863 46.019300370323556,-88.57535727888002 46.008959370313924,-88.597536278900677 46.015516370320029,-88.615502278917404 45.994120370300109,-88.643669278943634 45.993388370299428,-88.67738427897504 46.020144370324346,-88.703605278999461 46.018923370323208,-88.726409279020686 46.029581370333133,-88.773017279064106 46.021147370325281,-88.777480279068257 46.032614370335956,-88.793815279083475 46.036360370339445,-88.804397279093322 46.026804370330545,-88.925195279205823 46.073601370374128,-88.985301279261805 46.10039137039908,-89.099806279368451 46.145642370441223,-89.925136280137096 46.304025370588732,-90.111659280310803 46.34042937062263,-90.115177280314086 46.36515537064566,-90.141797280338878 46.393899370672429,-90.161391280357122 46.442380370717579,-90.211526280403817 46.50629537077711,-90.258401280447472 46.508789370779425,-90.26978528045808 46.522480370792181,-90.300181280486385 46.525051370794571,-90.302393280488445 46.544296370812496,-90.313708280498986 46.551563370819267,-90.385525280565858 46.539657370808172,-90.408200280586982 46.568610370835145,-90.018864280224392 46.678633370937604,-89.886252280100877 46.768935371021712,-89.791244280012393 46.824713371073656,-89.386718279635659 46.850208371097395,-89.214592279475355 46.923378371165541,-89.12518727939208 46.996606371233739,-88.994875279270715 46.997103371234203,-88.929688279210012 47.030926371265707,-88.884832279168236 47.104554371334274,-88.629500278930436 47.225812371447205,-88.61810427891983 47.131114371359018,-88.511215278820274 47.106506371336096,-88.512995278821933 47.032589371267257,-88.441164278755039 46.990734371228271,-88.445964278759504 46.928304371170128,-88.476523278787965 46.855151371102004,-88.446617278760115 46.799396371050079,-88.177827278509781 46.945890371186508,-88.189188278520362 46.900958371144668,-88.036685278378343 46.911865371154818,-87.90065427825165 46.909761371152868,-87.663766278031034 46.836851371084961,-87.37153927775887 46.507991370778683,-87.110679277515928 46.501473370772615,-87.006402277418815 46.536293370805041,-86.871382277293065 46.444359370719425,-86.759495277188861 46.486631370758793,-86.638220277075916 46.422263370698843,-86.462392276912169 46.561085370828138,-86.148109276619465 46.673053370932408,-86.096739276571626 46.655268370915849,-85.857536276348853 46.694815370952682,-85.503850276019449 46.674174370933457,-85.2300942757645 46.756785371010395,-84.954759275508067 46.770951371023585,-85.02697127557532 46.694339370952235,-85.018975275567882 46.549024370816902,-85.051655275598307 46.505576370776438,-85.016639275565694 46.476444370749306,-84.931320275486243 46.487843370759919,-84.803653275367338 46.444054370719144,-84.629815275205445 46.482943370755358,-84.572667275152213 46.407926370685487,-84.415967275006281 46.480658370753233,-84.31161427490909 46.48866937076069,-84.181646274788051 46.248720370537221,-84.273134274873257 46.207309370498649,-84.24703127484895 46.171447370465259,-84.119735274730388 46.176108370469592,-84.029578274646425 46.128943370425674,-84.061981274676612 46.094470370393566,-83.989501274609097 46.025985370329778,-83.901952274527574 46.005902370311077,-83.906460274531767 45.960239370268553,-84.11327227472438 45.978538370285591,-84.354485274949027 45.999190370304831,-84.501635275086059 45.978342370285411,-84.616845275193356 46.038230370341182,-84.689022275260584 46.035918370339033,-84.731732275300359 45.855679370171174,-84.851100275411525 45.89063637020373,-85.061629275607601 46.02475137032863,-85.378243275902463 46.100047370398755,-85.50954627602475 46.101911370400494,-85.655381276160568 45.972870370280319,-85.859844276350998 45.969469370277146)),((-83.854680274483542 46.014031370318648,-83.801105274433652 45.988412370294789,-83.756420274392028 46.027338370331037,-83.673592274314885 46.036192370339293,-83.680314274321148 46.071794370372444,-83.732448274369702 46.084108370383916,-83.649887274292809 46.103971370402412,-83.589498274236576 46.088518370388016,-83.533991274184871 46.011790370316561,-83.473189274128245 45.987547370293981,-83.516159274168274 45.925714370236399,-83.579813274227547 45.917501370228749,-83.629705274274016 45.95359637026236,-83.804881274437165 45.936764370246692,-83.852810274481797 45.997449370303201,-83.885891274512602 45.970852370278436,-83.854680274483542 46.014031370318648)),((-86.834829277259018 41.765504366361895,-86.617592277056701 41.907448366494094,-86.498833276946101 42.126446366698055,-86.374278276830097 42.249421366812584,-86.284980276746936 42.422324366973612,-86.21785427668442 42.774825367301901,-86.273837276736558 43.121045367624347,-86.463201276912912 43.475166367954145,-86.541301276985649 43.663187368129257,-86.447811276898577 43.772665368231216,-86.404345276858095 43.766642368225604,-86.434101276885812 43.781458368239399,-86.428814276880885 43.820123368275418,-86.459548276909516 43.950184368396542,-86.43814727688958 43.945592368392269,-86.518602276964515 44.053619368492875,-86.386423276841413 44.183204368613559,-86.271954276734803 44.351228368770045,-86.238038276703222 44.522273368929341,-86.258627276722393 44.700731369095543,-86.108484276582558 44.734442369126938,-86.082918276558743 44.777929369167438,-86.097964276572768 44.85061236923513,-86.067454276544353 44.898257369279506,-85.795756276291314 44.985974369361202,-85.61021527611851 45.196527369557288,-85.565514276076883 45.18056036954242,-85.653006276158365 44.958362369335482,-85.638039276144426 44.778435369167909,-85.526081276040159 44.76316236915369,-85.451351275970552 44.860540369244376,-85.384869275908642 45.010603369384135,-85.390244275913645 45.211593369571318,-85.373253275897824 45.273541369629015,-85.305475275834695 45.320383369672641,-85.092862275636691 45.370225369719059,-84.985893275537066 45.373178369721813,-84.921674275477258 45.409899369756005,-85.081815275626397 45.464650369807003,-85.120447275662372 45.569779369904907,-85.078019275622864 45.630185369961168,-84.983412275534761 45.68371337001102,-84.972038275524156 45.737745370061333,-84.724186275293334 45.780304370100978,-84.465275275052207 45.653637369983002,-84.321458274918257 45.665607369994156,-84.205560274810324 45.630905369961837,-84.135229274744816 45.571343369906359,-84.105907274717509 45.498749369838755,-83.922892274547067 45.491773369832259,-83.782809274416607 45.409449369755592,-83.712318274350963 45.412394369758331,-83.592363274239233 45.349502369699763,-83.495832274149336 45.360802369710285,-83.489598274143532 45.328937369680602,-83.394019274054514 45.272907369628427,-83.420761274079425 45.25718236961378,-83.398695274058866 45.213641369573224,-83.312707273978788 45.098620369466104,-83.444441274101479 45.052773369423406,-83.43397227409173 45.011128369384622,-83.46490327412053 44.997883369372289,-83.429355274087428 44.926297369305615,-83.319724273985329 44.860646369244478,-83.280812273949081 44.703183369097829,-83.320036273985622 44.515460368922994,-83.356963274020003 44.335133368755052,-83.529150274180367 44.261274368686266,-83.56823727421677 44.170118368601372,-83.598404274244871 44.070493368508593,-83.704802274343962 43.997165368440292,-83.873615274501176 43.962842368408332,-83.918376274542865 43.916997368365635,-83.938121274561254 43.698283368161938,-83.699164274338699 43.599642368070079,-83.654615274297214 43.607420368077314,-83.530909274182008 43.7259433681877,-83.494248274147864 43.70284136816619,-83.466408274121932 43.745740368206143,-83.367163274029508 43.844452368298072,-83.326026273991189 43.940459368387486,-82.940154273631819 44.069959368508094,-82.805978273506867 44.033564368474195,-82.727902273434154 43.972506368417328,-82.618487273332249 43.787866368245375,-82.60573827332037 43.694568368158485,-82.503820273225458 43.172253367672042,-82.41983627314724 42.972465367485967,-82.471952273195782 42.898682367417251,-82.473238273196969 42.762896367290793,-82.518179273238829 42.634052367170796,-82.645877273357755 42.631728367168634,-82.634015273346705 42.669382367203703,-82.729806273435926 42.681226367214734,-82.820407273520303 42.635794367172423,-82.802361273503493 42.612926367151118,-82.88813827358338 42.495756367041999,-82.874907273571054 42.458067367006905,-82.929389273621794 42.363040366918398,-83.107588273787755 42.292705366852893,-83.193873273868121 42.115749366688092,-83.190066273864574 42.033979366611938,-83.482691274137096 41.725130366324294,-83.76395427439904 41.717042366316761,-83.868639274496545 41.715993366315786,-84.359208274953417 41.708039366308384,-84.384393274976873 41.707150366307552,-84.79037727535497 41.697494366298557,-84.788478275353214 41.760959366357667,-84.826008275388162 41.761875366358524,-85.193140275730073 41.762867366359444,-85.297209275827001 41.763581366360114,-85.65945927616437 41.762627366359219,-85.799227276294545 41.763535366360067,-86.06830227654514 41.76462836636108,-86.234565276699982 41.764864366361309,-86.525181276970642 41.765540366361932,-86.834829277259018 41.765504366361895)))'
        url_wkt = reverse_wkt(wkt)
        url = '/api/archive/usgs-cida-maurer/model/miroc3.2(medres)/scenario/sres-a1b/run/2/temporal/2000-01-01+2000-03-01/spatial/intersects+{url_wkt}/aggregate/false/variable/pr.shz'
        url = url.format(url_wkt=url_wkt)
        response = self.client.get(url)
        self.assertEqual(response.status_code,200)
        
    def test_triangle_url(self):
        ext = 'kml'
        sop = 'intersects'
        agg = 'false'
        url = '/api/archive/usgs-cida-maurer/model/miroc3.2(medres)/scenario/sres-a1b/run/2/temporal/2000-01-01+2000-03-01/spatial/{1}+polygon((-94+39.75,-93.75+39.75,-93.75+40,-94+39.75))/aggregate/{2}/variable/pr.{0}'.format(ext,sop,agg)
        response = self.client.get(url)
        with open(get_temp_path(suffix='.'+ext),'w') as f:
            f.write(response.content)
        self.assertEqual(response.status_code,200)

    def test_api_urls(self):
        '''tests a combination of resources and formats'''
        resources = [
            '/api/archives',
            '/api/archives/usgs-cida-maurer',
            '/api/scenarios',
            '/api/scenarios/sres-a1b',
            '/api/models',
            '/api/models/echam5-mpi-om',
            '/api/variables',
            '/api/variables/pr',
            '/api/simulations',
            '/api/simulations/1',
            '/api/aois',
        ]
        suffixes = [
            '',
            '/',
            '.html',
            '.json',
            #'.kml',
        ]
        for resource in resources:
            for suffix in suffixes:
                print 'testing: {0}{1}'.format(resource,suffix)
                response = self.client.get('{0}{1}'.format(resource,suffix))
                if response.status_code != 200:
                    print response.content
                self.assertEqual(response.status_code, 200)


    def test_simulations_with_query_string_filter(self):
        '''Test that a query string filter reduces # of records returned.'''
        resource = '/api/simulations'
        filter1 = '?variable=pr&variable=tas'
        filter2 = '?variable=tas'
        filter3 = '?variable=pr&variable=tas&model=ccsm3'
        filter4 = '?variable=pr&model=ccsm3'
        content_filter1 = self.client.get(resource + filter1).content
        content_filter2 = self.client.get(resource + filter2).content
        content_filter3 = self.client.get(resource + filter3).content
        content_filter4 = self.client.get(resource + filter4).content
        content_unfiltered = self.client.get(resource).content
        
        # filter 1 should not affect the results
        self.assertTrue(len(content_unfiltered) == len(content_filter1))
        # filter 2 should return one record
        self.assertTrue(len(content_unfiltered) > len(content_filter2))
        # filter 3 should return the same records as filter 2
        self.assertTrue(len(content_filter2) == len(content_filter3))
        # filter 4 should not return any records
        self.assertTrue(len(content_filter3) > len(content_filter4))
    
    def test_data_request_urls(self):
        '''tests that data request URLs work
        
        This tests many different combinations of:
        * output formats (CSV, Shapefile, GeoJSON)
        * spatial operations (intersects or clip)
        * aggregation
        '''
        exts = [
            'csv',
            'kcsv',
            'shz',
            'lshz',
            'geojson',
        ]
        drange = '2010-3-1+2010-4-30'
        polygon = '-96.25+38.7,-95.78+38.1,-95.9+39.1,-96.23+39.8,-96.25+38.7'
        sops = [
            'intersects',
            'clip',
        ]
        aggs = [
            'true',
            'false',
        ]
        cm = 'miroc3.2(medres)'
        scenario = 'sres-a1b'
        archive = 'usgs-cida-maurer'
        var = 'pr'
        run = 2
        
        base_url = ('/api'
                    '/archive/{archive}/model'
                    '/{cm}/scenario/{scenario}'
                    '/run/{run}'
                    '/temporal/{drange}'
                    '/spatial/{sop}+polygon(({polygon}))'
                    '/aggregate/{agg}'
                    '/variable/{variable}.{ext}')
                    
        for ext,sop,agg in itertools.product(exts,sops,aggs):
            
            print(ext,sop,agg)
            
            if not (sop=='intersects' and agg=='true'):
                url = base_url.format(ext=ext,
                                      drange=drange,
                                      polygon=polygon,
                                      sop=sop,
                                      agg=agg,
                                      cm=cm,
                                      scenario=scenario,
                                      archive=archive,
                                      variable=var,
                                      run=run)
    
                response = self.client.get(url)
    #            if response.status_code != 200:
    #                print response.content
                self.assertEqual(response.status_code, 200)
    
    def test_simple_metadata_request(self):
        '''tests that a simple metadata request URL works'''
    
        url = ('/api'
               '/archive/{archive}'
               '/model/{cm}'
               '/scenario/{scenario}'
               '/run/{run}'
               '/temporal/{drange}'
               '/spatial/{sop}+polygon(({polygon}))'
               '/aggregate/{agg}'
               '/variable/{variable}.{ext}'
               ).format(ext='meta',
                        drange='2000-01-01+2000-02-01',
                        polygon='-104+39.75,-103.75+39.75,-103.75+40,-104+39.75',
                        sop='intersects',
                        agg='false',
                        cm='miroc3.2(medres)',
                        scenario='sres-a1b',
                        archive='usgs-cida-maurer',
                        variable='pr',
                        run=2,
                )
        response = self.client.get(url)
        if response.status_code != 200:
            print response.content
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            'Scenario link :: http://testserver/api/scenarios/sres-a1b'
            in response.content
        )
    
    def test_simple_json_data_request(self):
        '''tests that a simple data request URLs works'''
    
        url = ('/api'
               '/archive/{archive}'
               '/model/{cm}'
               '/scenario/{scenario}'
               '/run/{run}'
               '/temporal/{drange}'
               '/spatial/{sop}+polygon(({polygon}))'
               '/aggregate/{agg}'
               '/variable/{variable}.{ext}'
               ).format(ext='json',
                        drange='2000-01-01+2000-02-01',
                        polygon='-104+39.75,-103.75+39.75,-103.75+40,-104+39.75',
                        sop='intersects',
                        agg='false',
                        cm='miroc3.2(medres)',
                        scenario='sres-a1b',
                        archive='usgs-cida-maurer',
                        variable='pr',
                        run=2,
                )
        response = self.client.get(url)
        if response.status_code != 200:
            print response.content
        self.assertEqual(response.status_code, 200)
    
    def test_simple_kml_data_request(self):
        '''tests that a simple KML data request works'''
    
        url = ('/api'
               '/archive/{archive}'
               '/model/{cm}'
               '/scenario/{scenario}'
               '/run/{run}'
               '/temporal/{drange}'
               '/spatial/{sop}+polygon(({polygon}))'
               '/aggregate/{agg}'
               '/variable/{variable}.{ext}'
               ).format(
                    ext='kml',
                    drange='2000-01-01+2000-03-01',
                    polygon='-94+39.75,-93.75+39.75,-93.75+40,-94+39.75',
                    sop='intersects',
                    agg='false',
                    cm='miroc3.2(medres)',
                    scenario='sres-a1b',
                    archive='usgs-cida-maurer',
                    variable='pr',
                    run=2,
                )
        response = self.client.get(url)
        if response.status_code != 200:
            print response.content
        self.assertEqual(response.status_code, 200)

    def test_simple_kmz_data_request(self):
        '''tests that a simple KMZ data request works'''
        
        url = ('/api'
               '/archive/{archive}'
               '/model/{cm}'
               '/scenario/{scenario}'
               '/run/{run}'
               '/temporal/{drange}'
               '/spatial/{sop}+polygon(({polygon}))'
               '/aggregate/{agg}'
               '/variable/{variable}.{ext}'
               ).format(
                    ext='kmz',
                    drange='2000-01-01+2000-02-01',
                    polygon='-104+39.75,-103.75+39.75,-103.75+40,-104+39.75',
                    sop='intersects',
                    agg='false',
                    cm='miroc3.2(medres)',
                    scenario='sres-a1b',
                    archive='usgs-cida-maurer',
                    variable='pr',
                    run=2,
                )
        response = self.client.get(url)
        if response.status_code != 200:
            print response.content
        self.assertEqual(response.status_code, 200)
    
    
    def test_simple_kmz_data_request_detroit(self):
        '''tests that a KMZ data request for the detroit area works'''
        
        poly = 'POLYGON ((-83.142128872393371 42.22762950293189,-83.142128872393371 42.22762950293189,-83.142128872393371 42.22762950293189,-83.17525346549246 42.146264960288335,-83.183237474281555 42.10817942968923,-83.206207648435296 42.098877671876693,-83.298992682615363 42.090066840170934,-83.311834276039889 42.127997341473161,-83.311679246743012 42.178976141929098,-83.537324388345468 42.207346503257341,-83.781598883789158 42.217061672528203,-83.811958787760631 42.256542466799203,-83.819684414388263 42.31865753841403,-83.721783413411316 42.330181382815113,-83.629179246744286 42.280572007814925,-83.565022956053923 42.279900214195123,-83.514896816730811 42.293258571942573,-83.536626756509534 42.417592068036825,-83.578381313801373 42.521849270185683,-83.510788540363592 42.634555569014253,-83.48949785025934 42.76317820898872,-83.414722052732998 42.774960435551279,-83.349790615558263 42.750465806644925,-83.301369798500787 42.784572251957556,-83.273593716143893 42.839349270186943,-83.25641130240686 42.837488918624445,-83.24819474967245 42.826662706059295,-83.221271328448381 42.790075791996642,-83.210083380857185 42.737753404301131,-83.056139289059701 42.732947496097978,-82.943381313798824 42.686955471358218,-82.887648281572041 42.689048366866032,-82.828091193355661 42.698866889001494,-82.808299119787875 42.684009914717578,-82.790083177404995 42.665148016931042,-82.787835252600289 42.663830267907599,-82.804423387365972 42.63892222754292,-82.783804490881522 42.612722276370931,-82.796000128902406 42.589131985029695,-82.840958624996333 42.568099677086906,-82.870104132808947 42.519007066409628,-82.883384975907973 42.441854152997863,-82.902014329749193 42.394001776695589,-82.925992194332622 42.375501613935086,-83.016503465491837 42.347131252606857,-83.063373989580555 42.32266246191665,-83.104818488278639 42.286437282880044,-83.142128872393371 42.22762950293189))'
        url = ('/api'
               '/archive/{archive}'
               '/model/{cm}'
               '/scenario/{scenario}'
               '/run/{run}'
               '/temporal/{drange}'
               '/spatial/{sop}+{polygon}'
               '/aggregate/{agg}'
               '/variable/{variable}.{ext}'
               ).format(
                    ext='shz',
                    drange='2000-01-01+2000-02-01',
                    polygon=reverse_wkt(poly),
                    sop='clip',
                    agg='false',
                    cm='miroc3.2(medres)',
                    scenario='sres-a1b',
                    archive='usgs-cida-maurer',
                    variable='pr',
                    run=2,
                )
        response = self.client.get(url)
        if response.status_code != 200:
            print response.content
        self.assertEqual(response.status_code, 200)
    
    
    def test_clip_of_nonaggregated_geometries(self):
        '''tests that clipped geometries differ from intersected geometries
        for non-aggregated geometries
        '''
        from lxml import etree
        
        url_template = (
            '/api'
            '/archive/usgs-cida-maurer'
            '/model/miroc3.2(medres)'
            '/scenario/sres-a1b'
            '/run/2'
            '/temporal/2000-01-01+2000-02-01'
            '/spatial/{operation}+polygon((-104+39,+-103+39,+-103+40,+-104+39))'
            '/aggregate/false'
            '/variable/pr.kml'
        )
        url_clip = url_template.format(operation='clip')
        response_clip = self.client.get(url_clip)
        doc_clip = etree.fromstring(response_clip.content)
        folder_kml_string_clip = etree.tostring(
            doc_clip.find('.//{http://www.opengis.net/kml/2.2}Folder')
        )
        
        url_intersects = url_template.format(operation='intersects')
        response_intersects = self.client.get(url_intersects)
        doc_intersects = etree.fromstring(response_intersects.content)
        folder_kml_string_intersects = etree.tostring(
            doc_intersects.find('.//{http://www.opengis.net/kml/2.2}Folder')
        )
        self.assertNotEqual(
            folder_kml_string_clip,
            folder_kml_string_intersects
        )
    
    def test_query_form(self):
        '''Creates a query form'''
        response = self.client.get(
            '/api'
            '/archive/usgs-cida-maurer'
            '/model/ccsm3'
            '/scenario/sres-b1'
            '/variable/tas'
            '/run/2'
            '/query.html'
        )
        if response.status_code != 200:
                print response.content
        self.assertEqual(response.status_code, 200)
    
#    def OLD_test_urls(self):
#
#        ## list of extensions to test
#        exts = [
#                'shz',
#                'geojson',
#                'json',
#                'html'
#                ]
#        ## date ranges to test
#        dranges = [
#                   '2011-2-15',
#                   '2011-01-16+2011-3-16',
#                   ]
#        ## polygons intersections to test
#        polygons = [
##                    '11.5+3.5,12.5+3.5,12.5+2.5,11.5+2.5',
##                    '10.481+5.211,10.353+0.698,13.421+1.533,13.159+4.198',
#                    '18.746123371481431+80.295526668209391,18.261118856852192+82.963051498670211,0.073449558255717+86.358083101074868,36.206285898134041+86.843087615704121,66.5190680624615+81.023033440153256,106.531940519373734+64.532879942759109,181.707640286905814+82.478046984040958,205.715363761053169+65.017884457388362,280.163556756641356+78.598010867007048,288.651135762653098+47.557721930735738,293.986185423574739+70.110431860995362,294.956194452833188+49.982744503881918,302.23126217227184+74.717974749973138,300.291244113754829+43.192681299072575,355.824261038802774+86.358083101074868,356.551767810746583+-86.061021849619692,225.35804660353736+-83.150994761844245,145.089799432398252+-19.857905602728522,106.774442776688346+-54.778230656033756,93.436818624384273+-38.045574901324997,59.001498085708292+-80.483469931383439,22.626159488515341+-77.33094058629338,10.258544365469746+-69.328366094910933,22.868661745829968+-31.25551169651564,24.808679804346923+8.272356245767355,10.986051137413604+33.4925910064878,18.746123371481431+80.295526668209391',
#                    '71.009248245704413+28.048816528798497,84.841541328399558+26.255741499560216,87.40307708445421+14.984984172919738,82.792312723555824+7.300376904755765,73.826937577364532+5.763455451122965,64.09310170435684+9.093451933994018,63.836948128751374+20.364209260634524',
#                    ]
#        ## spatial operations
#        sops = [
#                'intersects',
#                'clip'
#                ]
#        ## aggregation
#        aggs = [
#                'true',
#                'false'
#                ]
#        ## climate models
#        cms = [
#               'bccr-bcm2.0'
#               ]
#        ## scenarios
#        scenarios = [
#                     '1pctto2x',
#                     ]
#        ## archives
#        archives = [
#                    'cmip3',
#                    ]
#        ## variables
#        variables = [
#                     'ps',
#                     ]
#        
#        base_url = ('/api/archive/{archive}/model/{cm}/scenario/{scenario}/'
#                    'temporal/{drange}/spatial/{sop}+polygon'
#                    '(({polygon}))/aggregate/{agg}/'
#                    'variable/{variable}.{ext}')
#        
##        for ext,drange,polygon,sop,agg,cm,scenario,archive,variable in itertools.product(exts,dranges,polygons,sops,aggs,cms,scenarios,archives,variables):
##            print ext,drange,'polygon index: '+str(polygons.index(polygon)),sop,agg,cm,scenario,archive,variable,'\n'
##            url = base_url.format(ext=ext,drange=drange,polygon=polygon,sop=sop,agg=agg,cm=cm,scenario=scenario,archive=archive,variable=variable)
##            response = self.client.get(url)
##            self.assertTrue(response.content != None)
##            self.assertEqual(response.status_code,200)
#
#        dranges = [
##                   '1950-5-15',
#                   '1950-9-1+1951-11-30'
#                   ]
#        polygons = [
#                    '-105.810709477731606+41.745763941079858,-104.480587924505272+41.72442509263238,-104.480587924505272+41.72442509263238,-103.634146936088499+41.639069698842455,-102.816157745601714+42.179653859511987,-101.279760657383051+42.641995575874084,-100.703611749301047+42.421494141916774,-100.390641972071322+41.994717172967142,-100.710724698783537+41.361664669025195,-101.407793748067931+40.586353175433366,-103.207369967138874+39.988865418903892,-103.925777864870753+39.66878269219167,-105.206108771719627+39.298909319101988,-106.60735981977092+39.462507157199347,-106.806522405280745+40.337399943546089,-106.642924567183385+41.240744527822798,-105.810709477731606+41.745763941079858',
#                    ]
#        archives = [
#                    'maurer07'
#                    ]
#        cms = [
#               'bccr-bcm2.0',
#               'cccma-cgcm3.1'
#               ]
#        scenarios = [
#                     'sresa1b',
#                     'sresa2'
#                     ]
#        variables = [
#                     'Prcp',
#                     ]
#        
#        for ext,drange,polygon,sop,agg,cm,scenario,archive,variable in itertools.product(exts,dranges,polygons,sops,aggs,cms,scenarios,archives,variables):
#            print ext,drange,'polygon index: '+str(polygons.index(polygon)),sop,agg,cm,scenario,archive,variable,'\n'
#            url = base_url.format(ext=ext,drange=drange,polygon=polygon,sop=sop,agg=agg,cm=cm,scenario=scenario,archive=archive,variable=variable)
#            response = self.client.get(url)
#            self.assertTrue(response.content != None)
#            self.assertEqual(response.status_code,200)
        
        
#class OpenClimateShpTests(NetCdfAccessTest):
#    
#    def get_object(self):
#        """Return an example OpenClimateShp object."""
#        
#        qs = SpatialGridCell.objects.all().order_by('row','col')
#        geom_list = qs.values_list('geom',flat=True)
##        geom_list = obj.geom) for obj in qs]
#        na = NetCdfAccessor(self.rootgrp,self.var)
#        dl = na.get_dict(geom_list)
#        path = get_temp_path('.shp')
#        shp = OpenClimateShp(path,dl)
#        return(shp)
#    
#    def test_write(self):
#        """Write a shapefile."""
#        
#        shp = self.get_object()
#        shp.write()
        
        
#class TestHelpers(TestCase):
#    
#    def test_parse_polygon_wkt(self):
#        """Test the parsing of the polygon query string."""
#        
#        actual = 'POLYGON ((30 10,10 20,20 40,40 40,30 10))'
#        
#        qs = ['POLYGON((30+10,10+20,20+40,40+40))',
#              'polygon((30+10,10+20,20+40,40+40))',
#              'polygon((30 10,10 20,20 40,40 40))']
#        
#        for q in qs: 
#            wkt = parse_polygon_wkt(q)
#            self.assertEqual(wkt,actual)


class TestFileUpload(TestCase):
    """Test URLs for uploading files."""

    def test_upload_shapefile(self):
        '''Tests uploading a shapefile'''
        with open('api/testdata/ne_ia_mi.zip') as f:
            response = self.client.post(
                '/api/aoi_upload.html',
                {'code': 'TESTCODE', 
                 'desc': 'sample description.',
                 'uid_field': "objectid",
                 'filefld': f},
            )
        self.assertEqual(response.status_code, 302)
    
    def test_upload_shapefile_bad_code(self):
        '''Tests uploading a shapefile'''
        with open('api/testdata/ne_ia_mi.zip') as f:
            response = self.client.post(
                '/api/aoi_upload.html',
                {'code': 'argh!^^#$',
                 'desc': 'sample description.',
                 'uid_field': "objectid",
                 'filefld': f},
            )
        self.assertTrue('The AOI code provided is invalid' in response.content)

    def test_upload_kml(self):
        '''Tests uploading a KML file'''
        with open('api/testdata/testfile.kml') as f:
            response = self.client.post(
                '/api/aoi_upload.html',
                {'code': 'TESTCODE',
                 'desc': 'sample description.',
                 #'objectid': None,
                 'filefld': f},
            )
        self.assertEqual(response.status_code, 302)


if __name__ == '__main__':
    unittest.main()

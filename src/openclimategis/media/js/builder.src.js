/*global Ext, google*/
var App;
/*
Ext.require([
    'Ext.Component',
    'Ext.container.Viewport',
    'Ext.data.Model',
    'Ext.data.reader.Json',
    'Ext.data.Store',
    'Ext.data.TreeStore',
    'Ext.form.FieldContainer',
    'Ext.form.field.*',
    'Ext.Panel',
    'Ext.toolbar.Toolbar',
    'Ext.tree.*'
    ]);
*/
///////////////////////////////////////////////////////////////////// Overrides
Ext.define('App.ui.BaseField', {
    override: 'Ext.form.field.Base',
    initialize: function() {
        this.callOverridden(arguments);
        },
    labelWidth: 120,
    triggerAction: 'all',
    isQueryParam: true, // NOTE: All form fields are assumed to represent query parameters
    listeners: {
        change: function(field, newValue) { // Third argument is "oldValue"
            if (this.isQueryParam) {
                this.findParentByType('form').fireEvent('change', {
                    field: field,
                    newValue: newValue
                    });
                } // eo if
            } // eo change
        } // eo listeners
    });
Ext.define('App.ui.BaseContainer', {
    override: 'Ext.container.Container',
    initialize: function() {
        this.callOverridden(arguments);
        },
    getValues: function() {
        var results = {},
            fields = this.query("{getValue}"); // Assumption is only fields have getValue member
        Ext.each(fields, function(i) {
            results[i.name] = i.getValue();
            });
        return results;
        }
    });
Ext.define('App.ui.Toolbar', {
    override: 'Ext.toolbar.Toolbar',
    initialize: function() {
        this.callOverridden(arguments);
        },
    height: 28
    });
//////////////////////////////////////////////////////////////////////// Models
Ext.define('App.api.AreaOfInterest', {
    extend: 'Ext.data.Model',
    fields: ['uid_field', 'code', {name: 'id', type: 'int'}, 'desc']
    });
Ext.define('App.api.Archive', {
    extend: 'Ext.data.Model',
    fields: ['code', {name: 'id', type: 'int'}, 'name', 'url', 'urlslug']
    });
Ext.define('App.api.Function', {
    extend: 'Ext.data.Model',
    fields: [
        'text',
        {name: 'leaf', type: 'boolean'},
        {name: 'children', type: 'auto'},
        'value',
        'desc'
        ],
    hasFormatString: function() {
        if (this.get('desc').indexOf('{0}') > 0) {
            return true;
            }
        return false;
        },
    singleValued: function() {
        if (this.get('desc').indexOf('{1}') > 0) {
            return false;
            }
        return true;
        },
    getComponents: function() {
        var raw = this.get('desc');
        if (this.singleValued()) {
            if (raw.indexOf('{0}') > 0) { // Clip to the {0} placeholder
                this.set('first', raw.substr(0, raw.indexOf('{0}')));
                }
            return [
                {
                    xtype: 'textfield',
                    fieldLabel: this.get('first'),
                    labelAlign: 'top',
                    labelSeparator: '',
                    isQueryParam: false,
                    vtype: 'numeric'
                    }
                ]; // eo return
            } // eo if
        else {
            this.set('first', raw.substr(0, raw.indexOf('{0}')));
            this.set('second', raw.substring(raw.indexOf('{0}') + 3, raw.indexOf('{1}')));
            return [
                {
                    xtype: 'inlinetextfield',
                    fieldLabel: this.get('first'),
                    name: 'first',
                    isQueryParam: false,
                    vtype: 'numeric'
                    },
                {
                    xtype: 'inlinetextfield',
                    fieldLabel: this.get('second'),
                    name: 'second',
                    isQueryParam: false,
                    vtype: 'numeric'
                    }
                ]; // eo return
            } // eo else
        } // eo getComponents
    }, function() {
        Ext.apply(Ext.form.field.VTypes, {
            numeric: function(val) {
                var re = /^\d*\.?\d*$/; // Matches numbers only
                return re.test(val);
                },
            numericText: 'Only numeric values are allowed'
            });
        });
Ext.define('App.api.Model', {
    extend: 'Ext.data.Model',
    fields: ['urlslug', 'code', 'organization', 'id', 'name', 'comments']
    });
Ext.define('App.api.Scenario', {
    extend: 'Ext.data.Model',
    fields: ['urlslug', 'code', 'description', 'id', 'name']
    });
Ext.define('App.api.Variable', {
    extend: 'Ext.data.Model',
    fields: ['ndim', 'units', 'urlslug', 'code', 'description', 'id', 'name']
    });
/////////////////////////////////////////////////////////////////////// Classes
Ext.define('App.ui.Indicator', {
    extend: 'Ext.Component',
    alias: 'widget.indicator',
    baseCls: 'error-state',
    validText: 'Form is valid',
    invalidText: 'Form has errors',
    flex: 1,
    initComponent: function() {
        this.addEvents('valid', 'invalid');
        },
    setValid: function(valid) {
        if (valid) {
            this.addCls(this.baseCls + '-valid');
            this.removeCls(this.baseCls + '-invalid');
            this.update(this.validText);
            this.fireEvent('valid');
            } else {
            this.addCls(this.baseCls + '-invalid');
            this.removeCls(this.baseCls + '-valid');
            this.update(this.invalidText);
            this.fireEvent('invalid');
            }
        },
    listeners: {
        afterrender: function() {
            this.setValid(false); // Request URL is not complete to begin with
            this.fireEvent('invalid');
            }
        }
    });
Ext.define('App.ui.ApiComboBox', {
    extend: 'Ext.form.field.ComboBox',
    alias: 'widget.apicombo',
    queryMode: 'remote',
    valueField: 'urlslug',
    displayField: 'urlslug',
    onLoad: function(store) { // Arguments: [store, records, success]
        // Set the field to the value of the first record, based on store's valueField
        this.setValue(store.data.items[0].data[this.valueField]);
        }
    });
Ext.define('App.ui.OkButton', {
    extend: 'Ext.button.Button',
    alias: 'widget.ok',
    width: 69,
    text: 'OK',
    style: {margin: '5px 3px 0'},
    handler: function() {
        // The callback expects the button "type" (e.g. 'ok') and the form values
        if (!this.callback(this.text.toLowerCase(), this.ownerCt.getValues())) {
            this.findParentByType('window').close();
            }
        }
    });
Ext.define('App.ui.CancelButton', {
    extend: 'Ext.button.Button',
    alias: 'widget.cancel',
    width: 69,
    text: 'Cancel',
    style: {margin: '5px 3px 0'},
    handler: function() {
        // The callback expects the button "type" (e.g. 'ok') and its value
        if (!this.callback(this.text.toLowerCase(), this.ownerCt.getValues())) {
            this.findParentByType('window').close();
            }
        }
    });
Ext.define('App.ui.InlineTextField', {
    extend: 'Ext.form.field.Text',
    alias: 'widget.inlinetextfield',
    labelSeparator: '',
    labelAlign: 'top',
    fieldStyle: {width: 80}
    });
Ext.define('App.ui.MarkupComponent', { // No ExtJS fluff
    extend: 'Ext.Component',
    alias: 'widget.markup',
    frame: false,
    border: 0
    }); // No callback (third argument)
Ext.define('App.ui.Container', { // Children should only be panels
    extend: 'Ext.Panel',
    alias: 'widget.container',
    resizable: true
    }); // No callback (third argument)
Ext.define('App.ui.NestedPanel', { // Padded bodies
    extend: 'Ext.Panel',
    alias: 'widget.nested',
    resizable: true,
    bodyPadding: 7
    }); // No callback (third argument)
Ext.define('App.ui.MapPanel', {
    extend: 'Ext.Panel',
    alias: 'widget.mappanel',
    overlays: [], // Initialize the overlays holder
    initComponent : function(){
        var config = {
            layout: 'fit',
            mapConfig: {
                center: new google.maps.LatLng(42.30220, -83.68952),
                zoom: 8,
                type: google.maps.MapTypeId.ROADMAP
                }
            };
        Ext.applyIf(this, config);
        // Event overlaycomplete - An overlay has finished being drawn
        // Event sketchcomplete - An overlay has finished being drawn (for instance listeners)
        this.addEvents('change', 'mapready', 'overlaycomplete', 'sketchcomplete');
        this.callParent();
        },
    /**
     * Draws a feature with a given well-known text (WKT) representation on the map.
     * @param   text    {String}    The well-known text string geometry
     * @return          {google.maps.Polygon}
     */
    addWktPolygon: function(text) {
        var geometry = App.decodeWkt(text), polygon;
        this.clearOverlays();
        polygon = new google.maps.Polygon({
            editable: (function() {
                return false; // TODO Make it so that edits to AOI geometry are reflected in the URL
                /* Still a good limit for improved performance:
                if (geometry.multipolygon) {return (geometry.multipolygon.length < 25);}
                else if (geometry.polygon) {return (geometry.polygon.length < 25);}
                */
                }()), // Execute immediately
            paths: (function() {
                var paths = [];
                if (geometry.polygon) {
                    Ext.each(geometry.polygon, function(i) {
                        paths.push(new google.maps.LatLng(i.lat, i.lng));
                        });
                    }
                if (geometry.multipolygon) {
                    Ext.each(geometry.multipolygon, function(i) {
                        paths.push(new google.maps.LatLng(i.lat, i.lng));
                        });
                    }
                return paths;
                }()) // Execute immediately
            });
        this.overlays.push(polygon);
        this.gmap.setCenter(polygon.getPath().getAt(0));
        polygon.setMap(this.gmap);
        },
    /**
     * Generates a WKT string from an MVCArray defining a polygon path
     * @param   path    {Array}     An Array of {lat, lng} objects
     * @return          {String}    A polygon string (e.g. 'polygon((...))')
     */
    pathToWktPolygon: function(path) {
        var coords = [];
        path.forEach(function(i) {
            coords.push({
                lng: i.lng(),
                lat: i.lat()
                });
            });
        coords.push({ // Push the first coordinate pair onto the end for closure of geometry
            lng: path.getAt(0).lng(),
            lat: path.getAt(0).lat()
            });
        return App.encodeWkt('polygon', coords);
        },
    /**
     * Generates a WKT string from latitude-longitude bounds (as from a rectangle)
     * @param   bounds  {google.maps.LatLngBounds}  Bounds of, presumably, a rectangle
     * @return          {Array}                     An Array of {lat, lng} objects
     */
    boundsToWktPolygon: function(bounds) {
        var coords, b = bounds;
        coords = [ // An array of the each of the corners
            {lat: b.getNorthEast().lat(), lng: b.getSouthWest().lng()}, // NW
            {lat: b.getNorthEast().lat(), lng: b.getNorthEast().lng()}, // NE
            {lat: b.getSouthWest().lat(), lng: b.getNorthEast().lng()}, // SE
            {lat: b.getSouthWest().lat(), lng: b.getSouthWest().lng()}, // SW
            {lat: b.getNorthEast().lat(), lng: b.getSouthWest().lng()}  // NW (again, for closure)
            ];        
        return App.encodeWkt('polygon', coords);
        },
    /**
     * Removes any overlay(s) from the map
     */
    clearOverlays: function() {
        if (this.overlays.length > 0) {
            Ext.each(this.overlays, function(i) {
                i.setMap(null);
                });
            }
        this.overlays.length = 0; // Empty the holder
        },
    listeners: {
        // When the container is rendered //////////////////////////////////////
        render: function() {
            this.body.mask(); // Mask labels will not be placed correctly so don't provide text
            },
        // Set up the map and listeners ////////////////////////////////////////
        afterrender: function() {
            var self = this,
                Type = google.maps.drawing.OverlayType;
            this.drawingManager = new google.maps.drawing.DrawingManager({
                rectangleOptions: {editable: true},
                polygonOptions: {editable: true},
                drawingControlOptions: {
                    drawingModes: [Type.RECTANGLE, Type.POLYGON]
                    }
                });
            this.gmap = new google.maps.Map(this.body.dom, {
                center: new google.maps.LatLng(42.30220, -83.68952),
                zoom: 8,
                mapTypeId: google.maps.MapTypeId.ROADMAP
                });
            this.drawingManager.setMap(this.gmap);
            // Listen for the 'overlaycomplete' event and pass it to the container
            google.maps.event.addListener(this.drawingManager, 'overlaycomplete', function(event) {
                self.fireEvent('overlaycomplete', {event: event});
                });
            // Listen for the 'tilesloaded' event as proxy indicator for 'mapready'
            google.maps.event.addListener(this.gmap, 'tilesloaded', function() {
                self.fireEvent('mapready');
                });
            },
        // When the map API is ready ///////////////////////////////////////////
        mapready: function() {
            this.body.unmask();
            },
        // When a new AOI is drawn /////////////////////////////////////////////
        overlaycomplete: function(args) {
            var geometry,
                that = this,
                Type = google.maps.drawing.OverlayType;
            this.fireEvent('sketchcomplete'); // Listened for in instances
            // Remove any existing overlay (only one allowed at a time
            this.clearOverlays();
            // Set the drawing mode to "pan" (the hand) so users can immediately edit
            this.drawingManager.setDrawingMode(null);
            // Polygon drawn
            if (args.event.type === Type.POLYGON) {
                geometry = this.pathToWktPolygon(args.event.overlay.getPath());
                // New vertex is inserted
                google.maps.event.addListener(args.event.overlay.getPath(), 'insert_at', function(n) {
                    that.fireEvent('change', {
                        wkt: that.pathToWktPolygon(that.overlays[0].getPath())
                        });
                    });
                // Existing vertex is removed (insertion is undone)
                google.maps.event.addListener(args.event.overlay.getPath(), 'remove_at', function(n) {
                    that.fireEvent('change', {
                        wkt: that.pathToWktPolygon(that.overlays[0].getPath())
                        });
                    });
                // Existing vertex is moved (set elsewhere)
                google.maps.event.addListener(args.event.overlay.getPath(), 'set_at', function(n) {
                    that.fireEvent('change', {
                        wkt: that.pathToWktPolygon(that.overlays[0].getPath())
                        });
                    });
                }
            // Rectangle drawn
            else if (args.event.type === Type.RECTANGLE) { 
                geometry = this.boundsToWktPolygon(args.event.overlay.getBounds());
                // Listen for the 'bounds_changed' event and update the geometry
                google.maps.event.addListener(args.event.overlay, 'bounds_changed', function() {
                    that.fireEvent('change', {
                        wkt: that.boundsToWktPolygon(that.overlays[0].getBounds())
                        });
                    });
                } // eo else if
            this.overlays.push(args.event.overlay); // Remember this overlay
            this.fireEvent('change', {wkt: geometry});
            },
        // When map feature geometry changes ///////////////////////////////////
        change: function(args) {
            this.findParentByType('form').fireEvent('change', {
                field: {
                    name: 'geometry',
                    isQueryParam: true
                    },
                newValue: args.wkt
                });
            }
        }
    }); // No callback (third argument)
Ext.define('App.ui.DateRange', {
    extend: 'Ext.form.FieldContainer',
    alias: 'widget.daterange',
    msgTarget: 'side',
    layout: 'hbox',
    outFormatEach: 'Y-m-d',
    outFormat: '{0}+{1}', // e.g. Output format is 'Y-m-d+Y-m-d'
    defaults: {
        width: 90,
        hideLabel: true,
        vtype: 'daterange',
        isQueryParam: false,
        listeners: {
            change: function() {
                if (this.prev()) {
                    if (this.prev().getValue() === null) {
                        return;
                        }
                    }
                else if (this.next()) {
                    if (this.next().getValue() === null) {
                        return;
                        }
                    }
                this.findParentByType('form').fireEvent('change', {
                    field: {
                        name: this.ownerCt.name || 'temporal',
                        isQueryParam: this.ownerCt.isQueryParam || true
                        },
                    newValue: this.ownerCt.getValue()
                    });
                }
            }
        },
    getValue: function() {
        var d1, d2, v = this.getValues();
        // The outFormatEach property should follow Ext.Date formatting
        d1 = Ext.util.Format.date(v.startDate, this.outFormatEach);
        d2 = Ext.util.Format.date(v.endDate, this.outFormatEach);
        return Ext.String.format(this.outFormat, d1, d2); // e.g. this.outFormat = '{0} to {1}'
        },
    items: [
        {
            xtype: 'datefield',
            name: 'startDate',
            itemId: 'date-start',
            endDateField: 'date-end',
            emptyText: 'Start',
            margin: '0 5 0 0'
            },
        {
            xtype: 'datefield',
            name: 'endDate',
            itemId: 'date-end',
            startDateField: 'date-start',
            emptyText: 'End'
            }
        ]
    }, function() { // Callback function
        Ext.apply(Ext.form.field.VTypes, {
            daterange: function(val, field) {
                var date = field.parseDate(val), start, end;
                if (!date) {
                    return false;
                    }
                if (field.startDateField) {
                    start = field.ownerCt.getComponent(field.startDateField);
                    if (!start.maxValue || (date.getTime() !== start.maxValue.getTime())) {
                        start.setMaxValue(date);
                        start.validate();
                        }
                    }
                else if (field.endDateField) {
                    end = field.ownerCt.getComponent(field.endDateField);
                    if (!end.minValue || (date.getTime() !== end.minValue.getTime())) {
                        end.setMinValue(date);
                        end.validate();
                        }
                    }
                /**
                 * Always return true since we're only using this vtype to set the
                 * min/max allowed values (these are tested for after the vtype test)
                 */
                return true;
                }
            });
        });
Ext.define('App.ui.TreePanel', {
    extend: 'Ext.tree.Panel',
    alias: 'widget.treepanel',
    rootVisible: true,
    getValue: function() {
        var stats = this.getChecked(),
            value = [];
        Ext.each(stats, function(i) {
            if (i.get('attrs')) {
                value.push(i.data);
                }
            else {value.push(i.get('value'));}
            });
        return value;
        },
    listeners: {
        beforeitemmousedown: function(view, rec) {
            var that = this, cb, prompt;
            cb = function(btn, values) {
                if (btn === 'cancel') {
                    rec.set('checked', !rec.get('checked'));
                    } 
                else {
                    Ext.iterate(values, function(k, v) {
                        if (!Ext.isNumeric(v)) {
                            Ext.Msg.alert('Invalid Value', 'You must enter a numeric value only.').setIcon(Ext.Msg.ERROR);
                            rec.set('checked', !rec.get('checked'));
                            } // eo if
                        else { // Item checked and successfully validated
                            // Adds the user-defined values to the record
                            rec.set('attrs', values);
                            that.fireEvent('checkchange', {
                                node: rec,
                                checked: true
                                });
                            }
                        }); // eo Ext.iterate()
                    } // eo else
                };
            // Is the box already checked?
            if (rec.get('checked')) { // Uncheck the box
                rec.set('checked', !rec.get('checked'));
                if (rec.get('attrs')) { // Clear the user-defined values
                    rec.set('attrs', undefined);
                    }
                this.fireEvent('checkchange', {
                    node: rec,
                    checked: false
                    });
                }
            else { // Check the box
                rec.set('checked', !rec.get('checked'));
                // Is inline formatting needed?
                if (rec.hasFormatString()) {
                    prompt = Ext.create('Ext.Window', {
                        title: 'Add Statistic: ' + rec.get('text'),
                        buttons: Ext.Msg.OKCANCEL,
                        closable: false,
                        animateTarget: this.header.getEl(),
                        modal: true,
                        bodyPadding: 10,
                        width: 250,
                        items: rec.getComponents(),
                        style: {textAlign: 'center'}
                        });
                    prompt.add([
                        {xtype: 'ok', callback: cb},
                        {xtype: 'cancel', callback: cb}
                        ]);
                    prompt.show();
                    }
                else {
                    this.fireEvent('checkchange', {
                        node: rec,
                        checked: true
                        });
                    }
                } // eo else
            } // eo beforemousedown
        } // eo listeners
    });
////////////////////////////////////////////////////////////////////////////////
Ext.application({
    name: 'App',
    /////////////////////////////////////////////////// Application Entry Point
    launch: function() {
        Ext.getBody().mask('Loading...');
        App.host = window.location.host || 'openclimategis.org';
        App.helpContents = (function() {
            var t;
            t = "<span class=\"help-title\">Welcome to OpenClimateGIS!</span>";
            t +="<br />This web application was designed to help users write API queries.";
            t +="You can use this form to parameterize your data request.";
            t +="The corresponding API query will be continuously updated at the bottom as you make your selections.<br /><br />";
            t +="<div class=\"help-topic-title\">Data Selection</div><div class=\"help-topic\">";
            t +="Specify the <b>data archive</b>, the <b>climate model</b>, the <b>emissions scenario</b>, <b>output variable</b>, the number of <b>runs</b>, and the <b>date range</b> for the model in this panel.";
            t +="</div>";
            t +="<div class=\"help-topic-title\">Temporal</div><div class=\"help-topic\">";
            t +="This panel contains a list of statistical functions that can be applied to the data, with or without a specified <b>grouping</b>. Click on an entry or check the box next to it to include it in your request. Unlike the entries under <b>Basic Statistics</b>, the <b>Thresholds</b> require you to specify values (e.g. between x and y).";
            t +="</div>";
            t +="<div class=\"help-topic-title\">Spatial</div><div class=\"help-topic\">";
            t +="The map panel is used for selecting an area-of-interest (AOI). You can use the drawing tools in the upper-left corner of the map to draw a rectangle or a polygon. Once you have drawn a rectangle or polygon, you can edit its vertices by clicking and dragging them. Start drawing a new polygon or rectangle elsewhere in the map and the old one will be removed (you can only use single polygon/rectangle geometry for now). You can also select pre-defined AOIs to use from the <b>Area-of-Interest (AOI)</b> drop-down menu in the map's top toolbar. You will see the AOI drawn on the map, but in the current version of the API Query Builder you cannot edit pre-defined AOIs.";
            t +="Click the <b>Clip Output to AOI</b> button to have the climate model results clipped to your AOI. Leaving this off (not toggled) will return climate data cells that are intersected by your AOI. Click the <b>Aggregate Geometries</b> button to treat multigeometries as singular.";
            t +="</div>";
            t +="<div class=\"help-topic-title\">Output Format</div><div class=\"help-topic\">";
            t +="You need to specify an <b>output format</b>, selected from the drop-down menu.";
            t +="</div>";
            t +="<div class=\"help-topic-title\">Data Request URL</div><div class=\"help-topic\">";
            t +="The result of your selections is displayed in the text box here as an API query URL. You can copy/paste this to save it or into a browser's navigation bar to execute it. Clicking the <b>Generate Data File</b> button will also execute your query. This button is disabled until the query is full parameterized; there are several required selections you have to make. You will know when your API query is full parameterized and ready to be executed when the indicator at the bottom of the screen changes from red to green and the text from \"URL Incomplete\" to \"Ready\".";
            t +="</div>";
            return t;
            }()); // Execute immediately
        /**
         * Encodes a WKT geometry from an array of coordinate pairs
         * @param   prefix  {String}    The type of WKT geometry (e.g. 'POLYGON')
         * @param   coords  {Array}     The coordinate pairs
         * @return          {String}    The corresponding WKT string
         */
        App.encodeWkt = function(prefix, coords) {
            var str = prefix + '((';
            coords.forEach(function(i, n, all) {
                str += i.lng.toFixed(5); // Longitude
                str += '+'; // Plus signs can be replaced with spaces
                str += i.lat.toFixed(5); // Latitude
                if (n < all.length-1) {str += ',';} // More coordinates?
                });
            str += '))';
            return str;
            };
        /**
         * Decodes a WKT geometry string into an object
         * @return      {Object}    The corresponding Javascript object
         */
        App.decodeWkt = function(text) {
            var geometry = {},
                prefix = Ext.String.trim(text.slice(0, text.indexOf('('))).toLowerCase(),
                remainder = text.slice(text.lastIndexOf('(')+1, text.indexOf(')'));
            geometry[prefix] = [];
            remainder.split(',').forEach(function(i) {
                geometry[prefix].push({
                    lng: Ext.String.trim(i).split(' ')[0],
                    lat: Ext.String.trim(i).split(' ')[1]
                    });
                });
            return geometry;
            };
        App.data = {
            aois: Ext.create('Ext.data.Store', {
                model: 'App.api.AreaOfInterest',
                proxy: {
                    type: 'ajax',
                    url: '/api/aois.json',
                    reader: 'json'
                    }
                }),
            archives: Ext.create('Ext.data.Store', {
                model: 'App.api.Archive',
                proxy: {
                    type: 'ajax',
                    url: '/api/archives.json',
                    reader: 'json'
                    }
                }),
            functions: Ext.create('Ext.data.TreeStore', {
                model: 'App.api.Function',
                proxy: {
                    type: 'ajax',
                    url: '/api/functions.json',
                    reader: 'json'
                    },
                sorters: [
                    {property: 'leaf', direction: 'ASC'},
                    {property: 'text', direction: 'ASC'}
                    ],
                root: {
                    text: 'Available Statistics',
                    expanded: true
                    }
                }),
            models: Ext.create('Ext.data.Store', {
                model: 'App.api.Model',
                proxy: {
                    type: 'ajax',
                    url: '/api/models.json',
                    reader: 'json'
                    }
                }),
            outputs: Ext.create('Ext.data.ArrayStore', {
                fields: ['value', 'text'],
                data: [
                    ['json', 'GeoJSON Text File'],
                    ['csv', 'Comma Separated Value'],
                    ['kcsv', 'Linked Comma Separated Value (zipped)'],
                    ['shz', 'ESRI Shapefile (zipped)'],
                    ['lshz', 'CSV-Linked ESRI Shapefile (zipped)'],
                    ['kml', 'Keyhole Markup Language'],
                    ['kmz', 'Keyhole Markup Language (zipped)'],
                    ['sqlite', 'SQLite3 Database (zipped)'],
                    ['nc', 'NetCDF']
                    ] 
                }),
            scenarios: Ext.create('Ext.data.Store', {
                model: 'App.api.Scenario',
                proxy: {
                    type: 'ajax',
                    url: '/api/scenarios.json',
                    reader: 'json'
                    }
                }),
            variables: Ext.create('Ext.data.Store', {
                model: 'App.api.Variable',
                proxy: {
                    type: 'ajax',
                    url: '/api/variables.json',
                    reader: 'json'
                    }
                })
            };
        //////////////////////////////////////////////////////////// Components
        App.viewport = Ext.create('Ext.container.Viewport', {
            id: 'viewport',
            layout: 'border',
            listeners: {
                afterrender: function() {
                    Ext.getBody().unmask();
                    }
                },
            items: [
                { // Banner
                    xtype: 'markup',
                    region: 'north',
                    height: 50,
                    style: {
                        background: '#000'
                        },
                    html: '<div id="branding"><a href="/">OpenClimateGIS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</a></div>'
                    },
                { // Form
                    xtype: 'form',
                    id: 'form-panel',
                    region: 'center',
                    layout: 'border',
                    listeners: {
                        change: function(args) { // Changes the API Request URL display
                            if (args.field.isQueryParam) {
                                Ext.getCmp('request-url').updateUrl(args.field.name, args.newValue);
                                } // eo if
                            }
                        },
                    items: [
                        { // Sidebar
                            xtype: 'container',
                            itemId: 'sidebar',
                            region: 'west',
                            width: 310,
                            border: 0,
                            layout: 'border',
                            items: [
                                { // Data selection
                                    xtype: 'nested',
                                    itemId: 'data-selection',
                                    title: 'Data Selection',
                                    region: 'north',
                                    height: 200
                                    },
                                { // Temporal selection
                                    xtype: 'treepanel',
                                    itemId: 'tree-panel',
                                    title: 'Temporal',
                                    region: 'center',
                                    store: App.data.functions,
                                    listeners: {
                                        // Event where statistical functions selected have changed
                                        checkchange: function() { // Arguments: node, checked
                                            var s = ''; // Create a query string
                                            Ext.each(this.getValue(), function(i, n, all) { // e.g. &stat=between(0,1)
                                                if (typeof(i) === 'object') {
                                                    s += i.value; // e.g. &stat=between
                                                    s += '('; // e.g. &stat=between(
                                                    Ext.each(Ext.Object.getValues(i.attrs), function(j, m, all) {
                                                        s += j; // e.g. &stat=between(0
                                                        if (m < all.length-1) {s += ',';}
                                                        });
                                                    s += ')'; // e.g. &stat=between(0,1)
                                                    }
                                                else {s += i;} // e.g. &stat=max
                                                if (n < all.length-1) {s += '+';}
                                                });
                                            Ext.getCmp('request-url').updateQuery('stat', s);
                                            }
                                        },
                                    tbar: [
                                        {
                                            xtype: 'combo',
                                            fieldLabel: 'Grouping Interval',
                                            name: 'grouping',
                                            labelWidth: 100,
                                            queryMode: 'local',
                                            value: '',
                                            valueField: 'value',
                                            width: 200,
                                            store: Ext.create('Ext.data.ArrayStore', {
                                                fields: ['text', 'value'],
                                                data: [
                                                    ['None', ''],
                                                    ['Year', 'year'],
                                                    ['Month', 'month'],
                                                    ['Day', 'day']
                                                    ] // data
                                                }) // eo store
                                            } // eo combo
                                        ] // eo tbar
                                    },
                                { // Output format
                                    xtype: 'nested',
                                    itemId: 'output-format',
                                    title: 'Output Format',
                                    region: 'south',
                                    height: 70
                                    }
                                ]
                            },
                        { // Spatial selection
                            xtype: 'mappanel',
                            itemId: 'map-panel',
                            id: 'map-panel',
                            title: 'Spatial',
                            region: 'center',
                            listeners: {
                                sketchcomplete: function() { // Clear the AOI selector
                                    this.down('#aoi').reset();
                                    }
                                },
                            tbar: [
                                {
                                    xtype: 'combo',
                                    fieldLabel: 'Area-of-Interest (AOI)',
                                    name: 'aoi',
                                    itemId: 'aoi',
                                    isQueryParam: true,
                                    store: App.data.aois,
                                    displayField: 'code',
                                    valueField: 'id',
                                    emptyText: '(None Selected)',
                                    style: {textAlign: 'right'},
                                    listeners: {
                                        select: function(c, records) {
                                            var code = records[0].get('code'),
                                                that = this;
                                            Ext.getCmp('request-url').updateUrl(this.name, code);
                                            Ext.Ajax.request({
                                                url: '/api/aois/' + code + '.json',
                                                callback: function(o, s, resp) { // Arguments: options, success, response
                                                    that.up('#map-panel').addWktPolygon(Ext.JSON.decode(resp.responseText).features[0].geometry);
                                                    }
                                                });
                                            }
                                        }
                                    },
                                ' ',
                                {
                                    xtype: 'button',
                                    disabled: true,
                                    text: 'Manage AOIs',
                                    iconCls: 'icon-app-edit'
                                    },
                                {
                                    xtype: 'button',
                                    text: 'Clip Output to AOI',
                                    name: 'clip',
                                    iconCls: 'icon-scissors',
                                    enableToggle: true,
                                    pressed: true, // Enabled by default; note this has to be set also in the default parameters object
                                    handler: function(b) {
                                        this.findParentByType('form').fireEvent('change', {
                                            field: {
                                                name: this.name,
                                                isQueryParam: true
                                                },
                                            newValue: b.pressed
                                            });
                                        }
                                    },
                                {
                                    xtype: 'button',
                                    text: 'Aggregate Geometries',
                                    name: 'aggregate',
                                    iconCls: 'icon-shape-group',
                                    enableToggle: true,
                                    pressed: true, // Enabled by default; note this has to be set also in the default parameters object
                                    handler: function(b) {
                                        this.findParentByType('form').fireEvent('change', {
                                            field: {
                                                name: this.name,
                                                isQueryParam: true
                                                },
                                            newValue: b.pressed
                                            });
                                        }
                                    },
                                '->',
                                {
                                    xtype: 'button',
                                    disabled: true,
                                    text: 'Save Sketch As AOI',
                                    iconCls: 'icon-disk'
                                    }
                                ] // eo items
                            },
                        { // Data request URL
                            xtype: 'nested',
                            itemId: 'request-url',
                            region: 'south',
                            title: 'Data Request URL',
                            height: 150,
                            layout: 'fit',
                            bbar: [
                                {
                                    xtype: 'button',
                                    id: 'query-run-btn',
                                    disabled: true,
                                    iconCls: 'icon-page-do',
                                    text: 'Generate Data File',
                                    handler: function() { // Open window for API call
                                        // Alternately: this.findParentByType('panel').items.getComponent('').getValue();
                                        window.open(Ext.getCmp('request-url').getValue(), 'call');
                                        }
                                    },
                                {
                                    xtype: 'progressbar',
                                    id: 'query-progress',
                                    width: 180
                                    },
                                ' ', // Spacer
                                {
                                    xtype: 'indicator',
                                    id: 'query-indicator',
                                    baseCls: 'ready-state',
                                    validText: 'Ready',
                                    invalidText: 'URL Incomplete',
                                    listeners: {
                                        valid: function() {
                                            Ext.getCmp('query-run-btn').enable();
                                            },
                                        invalid: function() {
                                            Ext.getCmp('query-run-btn').disable();
                                            }
                                        } // eo listeners
                                    } // eo indicator
                                ] // eo bbar
                            } // eo nested
                        ] // eo items
                    },
                { // Help
                    xtype: 'panel',
                    title: 'Help',
                    region: 'east',
                    width: 200,
                    resizable: true,
                    autoScroll: true,
                    bodyPadding: 5,
                    bodyCls: 'help-contents',
                    collapsed: true,
                    collapsible: true,
                    html: App.helpContents
                    }
                ] // eo items
            }); // eo Ext.create
        // Add items to the Data Selection panel ///////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('sidebar').getComponent('data-selection');
            p.add([
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Archive',
                    name: 'archive',
                    store: App.data.archives
                    },
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Climate Model',
                    name: 'model',
                    displayField: 'code',
                    store: App.data.models
                    },
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Emissions Scenario',
                    name: 'scenario',
                    displayField: 'code',
                    store: App.data.scenarios
                    },
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Variable',
                    name: 'variable',
                    displayField: 'name',
                    store: App.data.variables
                    },
                {
                    xtype: 'numberfield',
                    fieldLabel: 'Run',
                    name: 'run',
                    value: 1,
                    minValue: 1,
                    maxValue: 99
                    },
                {
                    xtype: 'daterange',
                    fieldLabel: 'Date Range',
                    name: 'temporal',
                    labelWidth: 80,
                    width: 290
                    }
                ]);
            }()); // Execute immediately
        // Add items to the Output Format panel ////////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('sidebar').getComponent('output-format');
            p.add([
                {
                    xtype: 'combo',
                    name: 'format',
                    width: 250,
                    queryMode: 'local',
                    valueField: 'value',
                    value: 'json',
                    store: App.data.outputs
                    }
                ]);
            }()); // Execute immediately
        // Add items to the Data Request URL panel /////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('request-url');
            p.insert(0, {
                xtype: 'textarea',
                name: 'query',
                id: 'request-url',
                isQueryParam: false,
                height: 80,
                editable: true,
                cls: 'updateable', // Indicates this field should flash when updated
                fieldBodyCls: 'shell', // Indicates this field's body text is code to be copy/pasted
                template: 'http://' + App.host + '/api/archive/{0}/model/{1}/scenario/{2}/run/{3}/temporal/{4}/spatial/{5}/aggregate/{6}/variable/{7}.{8}',
                // This is a holder for the parameter values
                values: {
                    archive: '{0}',
                    model: '{1}',
                    scenario: '{2}',
                    run: '{3}',
                    temporal: '{4}', 
                    spatial: '{5}', // This is 'clip+' + values.geometry when clip is true
                    aggregate: true, // Button is 'on' by default
                    variable: '{7}',
                    format: '{8}',
                    clip: true, // Button is 'on' by default
                    geometry: undefined // This is essentially the "spatial" parameter but irrespective of "clip"
                    },
                listeners: {
                    afterrender: function() {
                        Ext.apply(this.values, Ext.getCmp('form-panel').getValues());
                        this.url = 'http://' + App.host + '/api'; // Initialize base URL
                        this.query = ''; // Initialize query (none)
                        this.setValue(this.url + this.query);
                        },
                    change: function() { // Draw some attention to this box
                        Ext.getCmp('request-url').container.highlight();
                        this.animate({
                            duration: 400,
                            easing: 'backIn',
                            from: {opacity: 0.4},
                            to: {opacity: 1}
                            });
                        this.updateIndicator();
                        }
                    },
                /**
                 * Updates the "API Ready" indicator
                 */
                updateIndicator: function() {
                    var valid = true; // Assume all is well
                    Ext.iterate(this.values, function(k, v) { // Arguments: key, value, object
                        var fmt = /\{\d\}/;
                        // If a value is undefined or set to a format string
                        if (v === undefined || fmt.test(v)) {
                            valid = false; // You proved me wrong
                            }
                        });
                    Ext.getCmp('query-indicator').setValid(valid);
                    },
                /**
                 * Updates the GET request parameters passed in the API request URL
                 * e.g. resource?param=value
                 */
                updateQuery: function(name, value) {
                    var v = this.values;
                    this.values[name] = value;
                    // Can't use Ext.Object.toQueryString() because of unreadable HTML character encoding
                    this.query = (function() {
                        var x = '?';
                        if (v.stat) {x += 'stat=' + v.stat;}
                        if (v.stat && v.grouping) {x += '&';}
                        if (v.grouping) {x += 'grouping=' + v.grouping;}
                        return x;
                        }()); // Execute immediately
                    // Replace format strings characters in this.url with ???
                    this.setValue(this.url.replace(/\{\d\}/g, '???') + this.query);
                    return this.getValue(); // Return the updated URL
                    },
                /**
                 * Updates the pseudo-parameters encoded in the API request URL
                 * e.g. host/resource/type/id
                 */
                updateUrl: function(name, value) {
                    var v = this.values;
                    this.values[name] = value; // Set the updated parameter's value
                    if (name === 'grouping') {
                        this.updateQuery(name, value);
                        return;
                        }
                    if (name === 'geometry' || name === 'clip' || name === 'aoi') {
                        this.values.spatial = (function() {
                            var prefix = (v.clip) ? 'clip+' : 'intersects+';
                            // Either clip+ or intersects+ must precede the geometry
                            switch(name) {
                                // In each case, set the competing parameter to null
                                case 'aoi':
                                    v.geometry = null; // There can only be one!
                                    return prefix + v.aoi;
                                case 'geometry':
                                    v.aoi = null; // There can only be one!
                                    return prefix + v.geometry;
                                default:
                                    return prefix + (v.geometry || v.aoi);
                                }
                            }());
                        }
                    this.url = Ext.String.format(this.template, v.archive, v.model, v.scenario, v.run, v.temporal, v.spatial, v.aggregate, v.variable, v.format);
                    // Replace format strings with ??? and add the GET query parameters to the end
                    this.setValue(this.url.replace(/\{\d\}/g, '???') + this.query);
                    return this.getValue(); // Return the updated URL
                    } // eo updateUrl()
                });
            }()); // Execute immediately
        ////////////////////////////////////////////////////////////////////////
        } // eo launch()
    }); // eo Ext.application


